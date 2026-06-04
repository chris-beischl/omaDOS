import logging

import numpy as np

from omados.engine.cards import Cards, Suit
from omados.engine.modes import GameContract, GameModeFactory
from omados.engine.rules import get_playable_suits

from .base import BaseAgent, BaseInformationState

logger = logging.getLogger(__name__)


class RandomAgent(BaseAgent):
    def decide_bidding(self, hand: Cards) -> GameContract | None:
        contract = GameModeFactory.create_sauspiel(
            caller_id=self.player_id, suit=Suit.Eichel
        )
        playable_suits = get_playable_suits(hand, contract.trumpf_cards)
        if len(playable_suits) == 0:
            return None  # No playable suits, so pass

        suit = playable_suits[0]  # Just pick the first playable suit
        return GameModeFactory.create_sauspiel(caller_id=self.player_id, suit=suit)

    def play_card(self, state: BaseInformationState) -> int:
        legal_moves: Cards = state.legal_moves
        if legal_moves.cards.sum() == 0:
            print(f"Player {self.player_id} has no legal moves to play.")
        return int(np.random.choice(legal_moves.cards.nonzero().squeeze(dim=1)))
