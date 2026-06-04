import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from omados.engine.cards import Cards
from omados.engine.modes import GameContract
from omados.engine.tricks import Trick

logger = logging.getLogger(__name__)


@dataclass
class BiddingState:
    hand: Cards
    player_id: int
    dealer_id: int
    declarations: dict[int, bool]  # player_id -> wants_to_play (Phase 1 results)
    bid_history: list[tuple[int, GameContract | None]]  # (player_id, contract_or_fold)
    current_highest_contract: GameContract | None
    current_right_holder: int
    is_challenger: bool


@dataclass
class BaseInformationState:
    hand: Cards
    legal_moves: Cards
    last_trick: Trick | None
    current_trick: Trick
    contract: GameContract


class BaseAgent(ABC):
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.memory: list[dict[str, Any]] = []  # Agents can store previous tricks here

    # TODO: remove once proper bidding logic is implemented!
    @abstractmethod
    def decide_bidding(self, hand: Cards) -> GameContract | None:
        """Return a GameContract if you want to play, else None."""
        pass

    @abstractmethod
    def play_card(self, state: BaseInformationState) -> int: ...

    def observe_trick_results(self, trick: Trick, winner_id: int) -> None:
        """Called by the Env after every trick so the agent can learn/remember."""
        self.memory.append({"trick": trick, "winner": winner_id})

    @abstractmethod
    def wants_to_play(self, hand: Cards, state: BiddingState) -> bool: ...

    @abstractmethod
    def make_bid(self, hand: Cards, state: BiddingState) -> GameContract | None:
        # Challenger role: reveal minimum / raise, or None = fold
        ...

    @abstractmethod
    def wants_to_continue(self, hand: Cards, state: BiddingState) -> bool:
        # Defender role: "still want to play?" — no game type revealed
        ...

    @abstractmethod
    def declare_final_game(self, hand: Cards, state: BiddingState) -> GameContract:
        # Final binding announcement — suit + game type now revealed
        ...

    @abstractmethod
    def wants_to_give_contra(
        self, contract: GameContract, first_trick: Trick
    ) -> bool: ...

    @abstractmethod
    def wants_to_give_retour(
        self, contract: GameContract, first_trick: Trick
    ) -> bool: ...
