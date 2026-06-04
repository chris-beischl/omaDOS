import logging
from dataclasses import dataclass
from enum import Enum

from .cards import ASSE, HERZ, NO_CARDS, OBER, SUIT_MASKS, UNTER, Cards, Suit

logger = logging.getLogger(__name__)


class Spielart(Enum):
    Sauspiel = "Sauspiel"
    Wenz = "Wenz"
    Farbwenz = "Farbwenz"
    Geier = "Geier"
    Farbgeier = "Farbgeier"
    Solo = "Solo"
    Ramsch = "Ramsch"
    Sie = "Sie"


@dataclass
class GameContract:
    spielart: Spielart
    trumpf_cards: Cards
    rufsau_suit: Suit | None = None
    caller_id: int = -1

    def __post_init__(self) -> None:
        self.rufsau: Cards = (
            SUIT_MASKS[self.rufsau_suit] & ASSE
            if self.rufsau_suit is not None
            else NO_CARDS
        )


def get_trumpf_cards(spielart: Spielart, suit: Suit | None = None) -> Cards:
    if (
        spielart == Spielart.Sauspiel
        or spielart == Spielart.Ramsch
        or spielart == Spielart.Sie
    ):
        return OBER | UNTER | HERZ
    if spielart == Spielart.Wenz:
        return UNTER
    if spielart == Spielart.Farbwenz:
        assert suit is not None
        return UNTER | SUIT_MASKS[suit]
    if spielart == Spielart.Geier:
        return OBER
    if spielart == Spielart.Farbgeier:
        assert suit is not None
        return OBER | SUIT_MASKS[suit]
    if spielart == Spielart.Solo:
        assert suit is not None
        return OBER | UNTER | SUIT_MASKS[suit]

    raise ValueError(f"Unknown Spielart: {spielart}")


class GameModeFactory:
    @staticmethod
    def create_sauspiel(caller_id: int, suit: Suit) -> GameContract:
        # Standard: Obers, Unters, and the Heart suit are trumps
        # But in Sauspiel, the chosen 'suit' (Ace) defines the partner
        return GameContract(
            spielart=Spielart.Sauspiel,
            trumpf_cards=get_trumpf_cards(spielart=Spielart.Sauspiel),
            rufsau_suit=suit,
            caller_id=caller_id,
        )

    # Provides GameContract for both normal Wenz and Farbwenz (if suit is not None)
    @staticmethod
    def create_wenz(caller_id: int, suit: Suit | None = None) -> GameContract:
        spielart = Spielart.Wenz if suit is None else Spielart.Farbwenz
        return GameContract(
            spielart=spielart,
            trumpf_cards=get_trumpf_cards(spielart, suit),
            caller_id=caller_id,
        )

    @staticmethod
    def create_geier(caller_id: int, suit: Suit | None = None) -> GameContract:
        spielart = Spielart.Geier if suit is None else Spielart.Farbgeier
        return GameContract(
            spielart=spielart,
            trumpf_cards=get_trumpf_cards(spielart, suit),
            caller_id=caller_id,
        )

    @staticmethod
    def create_solo(caller_id: int, suit: Suit = Suit.Herz) -> GameContract:
        return GameContract(
            spielart=Spielart.Solo,
            trumpf_cards=get_trumpf_cards(Spielart.Solo, suit),
            caller_id=caller_id,
        )

    @staticmethod
    def create_ramsch() -> GameContract:
        return GameContract(
            spielart=Spielart.Ramsch, trumpf_cards=get_trumpf_cards(Spielart.Ramsch)
        )
