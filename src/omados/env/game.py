import torch

from omados.agents.base import BaseAgent
from omados.engine.cards import Cards
from omados.engine.modes import GameContract
from omados.engine.rules import determine_winner, get_legal_moves
from omados.engine.tricks import Trick


class SchafkopfEnv:
    def __init__(self, agents: list[BaseAgent]):
        self.agents = agents
        self.scores = torch.zeros(4)
        self.dealer_id = 0

    # TODO: handle Ramsch case
    def play_game(self) -> tuple[torch.Tensor, GameContract] | None:
        # 1. Deal (using your randperm logic)
        card_indices = torch.randperm(32)
        hands = [Cards(card_indices[i * 8 : (i + 1) * 8]) for i in range(4)]

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

        # 3. Playing (8 Tricks)
        current_leader = (self.dealer_id + 1) % 4

        for _ in range(8):
            # trick_cards = []
            trick = Trick(lead_player_id=current_leader)

            for i in range(4):
                pid = (current_leader + i) % 4

                # Rules engine generates legal moves
                # lead_card = trick_cards[0] if i > 0 else -1
                legal = get_legal_moves(hands[pid], trick.lead_card, contract)

                # Agent makes a choice
                obs = {
                    "hand": hands[pid],
                    "current_trick": trick,
                    "legal_moves": legal,
                    "contract": contract,
                }
                card_idx = self.agents[pid].play_card(obs)

                # Update state
                # trick_cards.append(card_idx)
                trick.add_card(card_idx)
                hands[pid].cards[card_idx] = False  # Remove card from hand

            # Determine winner and points
            winner_id = determine_winner(trick, contract)
            # winner_id = determine_winner(trick_cards, current_leader, contract)
            self.scores[winner_id] += trick.points

            # Notify agents so they can update their memory
            for a in self.agents:
                a.observe_trick_results(trick, winner_id)

            current_leader = winner_id

        return self.scores, contract
