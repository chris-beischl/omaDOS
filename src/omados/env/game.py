import logging

import torch

from omados.agents.base import BaseAgent, BaseInformationState, BiddingState
from omados.engine.cards import DECK, Cards
from omados.engine.modes import GameContract, Spielart
from omados.engine.rules import (
    determine_winner,
    get_legal_moves,
    get_teams,
    get_viable_spielarten,
    is_running_away,
)
from omados.engine.scoring import GameOutcome, determine_outcome, get_rewards
from omados.engine.tricks import Trick

logger = logging.getLogger(__name__)


class SchafkopfEnv:
    def __init__(self, agents: list[BaseAgent]):
        self.agents = agents
        self.scores = torch.zeros(4)
        self.dealer_id = 0

    # TODO: handle Ramsch case
    def play_game(
        self, hands: list[Cards] | None = None
    ) -> tuple[GameOutcome, float, float, GameContract, list[int], list[int]] | None:

        ran_away = [False] * 4  # Track if each player has 'run away' from Rufsau suit

        # 1. Deal (using your randperm logic)
        if hands is None:
            card_indices = torch.randperm(32)
            player_cards = torch.zeros((4, 32), dtype=torch.bool)
            for i in range(4):
                player_cards[i, card_indices[i * 8 : (i + 1) * 8]] = True
            hands = [Cards(player_cards[i]) for i in range(4)]

        # 2. Bidding (Simplified: first one who wants to play wins)
        contract = self._run_bidding(hands)

        if not contract:  # Everyone passed -> Ramsch (logic omitted for brevity)
            return None

        # TODO: Handle Ramsch case where no one bids and everyone plays against each
        # other
        player_team, opponent_team = get_teams(contract, hands)

        # 3. Playing (8 Tricks)
        current_leader = (self.dealer_id + 1) % 4

        tricks: list[Trick] = []
        tricks_per_player: dict[int, list[Trick]] = {pid: [] for pid in range(4)}
        game_scores = torch.zeros(4)
        for trick_id in range(8):
            logger.debug(f"Trick {trick_id + 1}")
            trick = Trick(lead_player_id=current_leader)

            for i in range(4):
                pid = (current_leader + i) % 4
                logger.debug(f"\tPlayer {pid} \tHand: {hands[pid]}")

                # Rules engine generates legal moves
                legal = get_legal_moves(
                    hands[pid], trick.lead_card, contract, ran_away[pid]
                )
                logger.debug(f"\t\t\tLegal Moves: {legal}")
                # Agent makes a choice

                last_trick = tricks[-1] if len(tricks) else None

                state = BaseInformationState(
                    hand=hands[pid],
                    legal_moves=legal,
                    last_trick=last_trick,
                    current_trick=trick,
                    contract=contract,
                )

                card_idx = self.agents[pid].play_card(state)

                is_running_away_flag = is_running_away(
                    player_hand=hands[pid],
                    lead_card_idx=trick.lead_card,
                    played_card_idx=card_idx,
                    contract=contract,
                    ran_away=ran_away[pid],
                )
                if is_running_away_flag:
                    ran_away[pid] = True
                    logger.debug(
                        f"\t\t\tPlayer {pid} is running away from Rufsau suit!"
                    )

                logger.debug(f"\t\t\tPlayed Card: {DECK[card_idx]}")

                # Update state
                trick.add_card(card_idx)
                hands[pid].cards[card_idx] = False  # Remove card from hand

            # Determine winner and points
            winner_id = determine_winner(trick, contract)
            logger.debug("")
            logger.debug(f"\tTrick: \t\t{[DECK[idx] for idx in trick.card_indices]}")
            logger.debug(f"\tWinner: \t{winner_id}")
            logger.debug(f"\tTrick Points: \t{trick.points}")
            game_scores[winner_id] += trick.points
            tricks_per_player[winner_id].append(trick)
            logger.debug("")
            logger.debug(f"\tGame Scores: \t{game_scores}")

            # Notify agents so they can update their memory
            for a in self.agents:
                a.observe_trick_results(trick, winner_id)

            current_leader = winner_id
            logger.debug("-" * 50)

        game_outcome = determine_outcome(
            scores=game_scores,
            player_team=player_team,
            tricks_per_player=tricks_per_player,
        )
        player_team_reward, opponent_team_reward = get_rewards(game_outcome)
        logger.debug(f"Outcome: {game_outcome}")
        logger.debug(
            f"Player team reward: {player_team_reward}, Opponent team reward:\
{opponent_team_reward}",
        )

        return (
            game_outcome,
            player_team_reward,
            opponent_team_reward,
            contract,
            player_team,
            opponent_team,
        )

    def _pid_game_order(self) -> list[int]:
        return [(self.dealer_id + 1 + i) % 4 for i in range(4)]

    def _run_bidding(self, hands: list[Cards]) -> GameContract | None:
        interested_players, bidding_state = self._phase1_declarations(hands)

        bidders = [pid for pid in self._pid_game_order() if interested_players[pid]]
        logger.debug(f"[Bidding] Phase 1: players {bidders} want to play")

        # Ramsch / Zusammenwerfen
        if len(bidders) == 0:
            logger.debug("[Bidding] No one wants to play → Ramsch")
            return None

        assert bidding_state is not None
        if len(bidders) == 1:
            winner = bidders[0]
        else:
            winner, bidding_state = self._phase2_negotiation(
                hands, interested_players, bidding_state
            )

        bidding_state.is_challenger = False
        viable_winner = get_viable_spielarten(hands[winner], bidding_state, 0)
        contract = self.agents[winner].declare_final_game(
            hands[winner], bidding_state, viable_winner
        )

        spielart_value = contract.spielart.value
        rufsau_suit = contract.rufsau_suit.value if contract.rufsau_suit else ""
        logger.debug(f"[Bidding] P{winner} declares: {spielart_value} ({rufsau_suit})")
        return contract

    def _phase1_declarations(
        self, hands: list[Cards]
    ) -> tuple[list[bool], BiddingState | None]:
        declarations: dict[int, bool] = {}
        interested_players = [False] * 4
        current_right_holder = -1

        bidding_state = None
        bidding_position = 0
        for pid in self._pid_game_order():
            bidding_state = BiddingState(
                hand=hands[pid],
                player_id=pid,
                dealer_id=self.dealer_id,
                declarations=declarations,
                bid_history=[],
                current_highest_spielart=None,
                current_right_holder=current_right_holder,
                is_challenger=False,
            )

            viable = get_viable_spielarten(hands[pid], bidding_state, bidding_position)
            declaration = self.agents[pid].wants_to_play(
                hands[pid], bidding_state, viable
            )
            declarations[pid] = declaration
            interested_players[pid] = declaration
            if declaration:
                current_right_holder = pid
                bidding_position += 1
            bidding_state.declarations = declarations
            logger.debug(
                f"[Phase 1] P{pid}: {'wants to play' if declaration else 'passes'}"
            )

        return interested_players, bidding_state

    def _phase2_negotiation(
        self,
        hands: list[Cards],
        interested_players: list[bool],
        bidding_state: BiddingState,
    ) -> tuple[int, BiddingState]:

        # compute order of bidders
        bidders = [pid for pid in self._pid_game_order() if interested_players[pid]]
        assert len(bidders) >= 2, (
            "For a negotiation, at least two people need intention\
            to play"
        )

        defender_idx = 0
        defender = bidders[defender_idx]
        winner = defender
        for challenger_idx, challenger in enumerate(bidders[1:], start=1):
            logger.debug(
                f"[Phase 2] P{defender} (defender) vs P{challenger} (challenger)"
            )
            winner, bidding_state = self._phase2_bidding(
                hands, challenger, defender, challenger_idx, defender_idx, bidding_state
            )
            logger.debug(f"[Phase 2] Winner: P{winner}")
            defender = winner
            defender_idx = defender_idx if winner == defender else challenger_idx

        return winner, bidding_state

    def _phase2_bidding(
        self,
        hands: list[Cards],
        challenger: int,
        defender: int,
        challenger_idx: int,
        defender_idx: int,
        bidding_state: BiddingState,
    ) -> tuple[int, BiddingState]:

        while True:
            # Challenger reveals minimum or folds
            bidding_state.is_challenger = True
            bidding_state.player_id = challenger
            viable_challenger = get_viable_spielarten(
                hands[challenger], bidding_state, challenger_idx
            )

            bid: Spielart | None = None

            if challenger == bidding_state.current_right_holder:
                bid = self.agents[challenger].make_bid(
                    hands[challenger], bidding_state, viable_challenger
                )
            else:
                bid = self.agents[challenger].make_bid_or_fold(
                    hands[challenger], bidding_state, viable_challenger
                )

            if bid is None:
                logger.debug(f"[Phase 2]   P{challenger} folds → P{defender} wins")
                bidding_state.bid_history.append((challenger, None))
                return defender, bidding_state

            logger.debug(f"[Phase 2]   P{challenger} bids: {bid.value}")
            bidding_state.bid_history.append((challenger, bid))
            bidding_state.current_highest_spielart = bid
            bidding_state.current_right_holder = challenger

            # Defender responds
            bidding_state.is_challenger = False
            bidding_state.player_id = defender
            viable_defender = get_viable_spielarten(
                hands[defender], bidding_state, defender_idx
            )
            wants_to_continue = self.agents[defender].wants_to_continue(
                hands[defender], bidding_state, viable_defender
            )

            if not wants_to_continue:
                logger.debug(f"[Phase 2]   P{defender} folds → P{challenger} wins")
                bidding_state.bid_history.append((defender, None))
                return challenger, bidding_state
            else:
                logger.debug(f"[Phase 2]   P{defender} continues")
                bidding_state.bid_history.append(
                    (defender, bidding_state.current_highest_spielart)
                )
                bidding_state.current_right_holder = defender
