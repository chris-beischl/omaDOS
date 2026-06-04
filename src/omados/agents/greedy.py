import logging

from omados.engine.cards import CARD_POINTS, Cards, Suit
from omados.engine.modes import GameContract, GameModeFactory
from omados.engine.rules import get_playable_suits

from .base import BaseAgent, BaseInformationState

logger = logging.getLogger(__name__)


class GreedyAgent(BaseAgent):
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
        masked_points = (CARD_POINTS.float() + 1) * legal_moves.cards.float()
        return int(masked_points.argmax().item())
