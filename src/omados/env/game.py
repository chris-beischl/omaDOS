import logging

import torch

from omados.agents.base import BaseAgent
from omados.engine.cards import DECK, Cards
from omados.engine.modes import GameContract
from omados.engine.rules import determine_winner, get_legal_moves, is_running_away
from omados.engine.tricks import Trick

logger = logging.getLogger(__name__)


class SchafkopfEnv:
    def __init__(self, agents: list[BaseAgent]):
        self.agents = agents
        self.scores = torch.zeros(4)
        self.dealer_id = 0

    # TODO: handle Ramsch case
    def play_game(self) -> tuple[torch.Tensor, GameContract] | None:

        ran_away = [False] * 4  # Track if each player has 'run away' from Rufsau suit

        # 1. Deal (using your randperm logic)
        card_indices = torch.randperm(32)
        player_cards = torch.zeros((4, 32), dtype=torch.bool)
        for i in range(4):
            player_cards[i, card_indices[i * 8 : (i + 1) * 8]] = True
        hands = [Cards(player_cards[i]) for i in range(4)]

        # 2. Bidding (Simplified: first one who wants to play wins)
        contract = None
        for i in range(4):
            pid = (self.dealer_id + 1 + i) % 4
            decision = self.agents[pid].decide_bidding(hands[pid])
            if decision:
                contract = decision
                break

        if not contract:  # Everyone passed -> Ramsch (logic omitted for brevity)
            return None

        logger.debug(contract)

        # 3. Playing (8 Tricks)
        current_leader = (self.dealer_id + 1) % 4

        for trick_id in range(8):
            logger.debug(f"Trick {trick_id + 1}, Leader")
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
                obs = {
                    "hand": hands[pid],
                    "current_trick": trick,
                    "legal_moves": legal,
                    "contract": contract,
                }
                card_idx = self.agents[pid].play_card(obs)

                is_running_away_flag = is_running_away(
                    player_hand=hands[pid],
                    lead_card_idx=trick.lead_card,
                    played_card_idx=card_idx,
                    contract=contract,
                    ran_away=ran_away[pid],
                )
                if is_running_away_flag:
                    ran_away[pid] = True

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
            self.scores[winner_id] += trick.points
            logger.debug("")
            logger.debug(f"\tScores: \t{self.scores}")

            # Notify agents so they can update their memory
            for a in self.agents:
                a.observe_trick_results(trick, winner_id)

            current_leader = winner_id
            logger.debug("-" * 50)

        return self.scores, contract
