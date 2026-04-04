from abc import ABC, abstractmethod
from omados.engine.cards import Cards

class BaseAgent(ABC):
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.memory = [] # Agents can store previous tricks here

    @abstractmethod
    def decide_bidding(self, hand: Cards):
        """Return a GameContract if you want to play, else None."""
        pass

    @abstractmethod
    def play_card(self, observation: dict) -> int:
        """
        observation contains: 'hand', 'current_trick', 'legal_moves', 'contract'
        Returns: card_index
        """
        pass

    def observe_trick_results(self, trick_cards: list[int], winner_id: int):
        """Called by the Env after every trick so the agent can learn/remember."""
        self.memory.append({'cards': trick_cards, 'winner': winner_id})