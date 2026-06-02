from abc import ABC, abstractmethod
from typing import Any

from omados.engine.cards import Cards
from omados.engine.modes import GameContract
from omados.engine.tricks import Trick


class BaseAgent(ABC):
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.memory: list[dict[str, Any]] = []  # Agents can store previous tricks here

    @abstractmethod
    def decide_bidding(self, hand: Cards) -> GameContract | None:
        """Return a GameContract if you want to play, else None."""
        pass

    @abstractmethod
    def play_card(self, observation: dict[str, Any]) -> int:
        """
        observation contains: 'hand', 'current_trick', 'legal_moves', 'contract'
        Returns: card_index
        """
        pass

    def observe_trick_results(self, trick: Trick, winner_id: int) -> None:
        """Called by the Env after every trick so the agent can learn/remember."""
        self.memory.append({"trick": trick, "winner": winner_id})
