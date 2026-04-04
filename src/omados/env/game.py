import torch
from omados.engine.cards import Cards, CARD_POINTS
from omados.engine.rules import get_legal_moves, determine_winner
from omados.agents.base import BaseAgent

class SchafkopfEnv:
    def __init__(self, agents: list[BaseAgent]):
        self.agents = agents
        self.scores = torch.zeros(4)
        self.dealer_id = 0

    def play_game(self):
        # 1. Deal (using your randperm logic)
        card_indices = torch.randperm(32)
        hands = [Cards(card_indices[i*8:(i+1)*8]) for i in range(4)]
        
        # 2. Bidding (Simplified: first one who wants to play wins)
        contract = None
        for i in range(4):
            pid = (self.dealer_id + 1 + i) % 4
            decision = self.agents[pid].decide_bidding(hands[pid])
            if decision:
                contract = decision
                break
        
        if not contract: # Everyone passed -> Ramsch (logic omitted for brevity)
            return

        # 3. Playing (8 Tricks)
        current_leader = (self.dealer_id + 1) % 4
        
        for t in range(8):
            trick_cards = []
            
            for i in range(4):
                pid = (current_leader + i) % 4
                
                # Rules engine generates legal moves
                lead_card = trick_cards[0] if i > 0 else -1
                legal = get_legal_moves(hands[pid], lead_card, contract)
                
                # Agent makes a choice
                obs = {
                    'hand': hands[pid],
                    'current_trick': trick_cards,
                    'legal_moves': legal,
                    'contract': contract
                }
                card_idx = self.agents[pid].play_card(obs)
                
                # Update state
                trick_cards.append(card_idx)
                hands[pid].cards[card_idx] = False # Remove card from hand
            
            # Determine winner and points
            winner_id = determine_winner(trick_cards, current_leader, contract)
            trick_points = sum([CARD_POINTS[c] for c in trick_cards])
            self.scores[winner_id] += trick_points
            
            # Notify agents so they can update their memory
            for a in self.agents:
                a.observe_trick_results(trick_cards, winner_id)
                
            current_leader = winner_id

        return self.scores, contract