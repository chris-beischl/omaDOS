import logging
import random

import numpy as np

from omados.engine.cards import Cards
from omados.engine.modes import SPIELART_RANK, GameContract, Spielart
from omados.engine.rules import get_available_games, get_available_spielarten
from omados.engine.tricks import Trick

from .base import BaseAgent, BaseInformationState, BiddingState

logger = logging.getLogger(__name__)

SAUSPIEL_PROBABILITY = 0.9


class RandomAgent(BaseAgent):
    # TODO: remove once proper bidding logic is implemented
    def decide_bidding(self, hand: Cards) -> GameContract | None:
        available = get_available_games(hand, self.player_id)
        if not available:
            return None
        return self._pick_game(available)

    def wants_to_play(
        self, hand: Cards, state: BiddingState, viable_spielarten: list[Spielart]
    ) -> bool:

        return random.random() > 0.5 and len(viable_spielarten) > 0

    def make_bid(
        self, hand: Cards, state: BiddingState, viable_spielarten: list[Spielart]
    ) -> Spielart:
        """Must return a valid bid — cannot fold. Called when obligated to play."""
        if not viable_spielarten:
            return self._pick_spielart(get_available_spielarten(hand))
        return self._pick_spielart(viable_spielarten)

    def make_bid_or_fold(
        self, hand: Cards, state: BiddingState, viable_spielarten: list[Spielart]
    ) -> Spielart | None:
        """Can return None to fold. Called when folding is allowed."""
        if not viable_spielarten or random.random() > 0.5:
            return None
        return self._pick_spielart(viable_spielarten)

    def wants_to_continue(
        self, hand: Cards, state: BiddingState, viable_spielarten: list[Spielart]
    ) -> bool:
        """Defender role: still want to play?"""
        return len(viable_spielarten) > 0 and random.random() < 0.5

    def declare_final_game(
        self, hand: Cards, state: BiddingState, viable_spielarten: list[Spielart]
    ) -> GameContract:
        available = get_available_games(hand, self.player_id)
        viable = [g for g in available if g.spielart in viable_spielarten]
        if not viable:
            viable = available
        return self._pick_game(viable)

    def wants_to_give_contra(self, contract: GameContract, first_trick: Trick) -> bool:
        return False

    def wants_to_give_retour(self, contract: GameContract, first_trick: Trick) -> bool:
        return False

    def play_card(self, state: BaseInformationState) -> int:
        legal_moves: Cards = state.legal_moves
        if legal_moves.cards.sum() == 0:
            logger.warning("Player %d has no legal moves to play.", self.player_id)
        return int(np.random.choice(legal_moves.cards.nonzero().squeeze(dim=1)))

    def _pick_spielart(self, available: list[Spielart]) -> Spielart:
        """90% chance to pick Sauspiel, 10% chance to pick randomly from all."""
        if Spielart.Sauspiel in available and random.random() < SAUSPIEL_PROBABILITY:
            return Spielart.Sauspiel
        return random.choice(available)

    def _pick_game(self, available: list[GameContract]) -> GameContract:
        """90% chance to pick a Sauspiel, 10% chance to pick randomly from all."""
        sauspiele = [g for g in available if g.spielart == Spielart.Sauspiel]
        if sauspiele and random.random() < SAUSPIEL_PROBABILITY:
            return random.choice(sauspiele)
        return random.choice(available)

    def _get_higher_spielarten(
        self, hand: Cards, current: Spielart | None
    ) -> list[Spielart]:
        """Returns available Spielarten strictly higher than the current."""
        available = get_available_spielarten(hand)
        if current is None:
            return available
        current_rank = SPIELART_RANK[current]
        return [s for s in available if SPIELART_RANK[s] > current_rank]

    def _get_higher_games(
        self, hand: Cards, current: GameContract | None
    ) -> list[GameContract]:
        """Returns available games strictly higher than the current highest contract."""
        available = get_available_games(hand, self.player_id)
        if current is None:
            return available
        current_rank = SPIELART_RANK[current.spielart]
        return [g for g in available if SPIELART_RANK[g.spielart] > current_rank]
