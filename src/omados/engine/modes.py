from dataclasses import dataclass

from .cards import ASSE, HERZ, NO_CARDS, OBER, SUIT_MASKS, UNTER, Cards, Suit


@dataclass
class GameContract:
    spielart: str
    trumpf_cards: Cards
    rufsau_suit: Suit | None = None
    caller_id: int = -1

    def __post_init__(self) -> None:
        self.rufsau: Cards = (
            SUIT_MASKS[self.rufsau_suit] & ASSE
            if self.rufsau_suit is not None
            else NO_CARDS
        )


class GameModeFactory:
    @staticmethod
    def create_sauspiel(caller_id: int, suit: Suit) -> GameContract:
        # Standard: Obers, Unters, and the Heart suit are trumps
        # But in Sauspiel, the chosen 'suit' (Ace) defines the partner
        return GameContract(
            spielart="Sauspiel",
            trumpf_cards=OBER | UNTER | HERZ,  # Herz is standard trump in Sauspiel
            rufsau_suit=suit,
            caller_id=caller_id,
        )

    # Provides GameContract for both normal Wenz and Farbwenz (if suit is not None)
    @staticmethod
    def create_wenz(caller_id: int, suit: Suit | None = None) -> GameContract:
        if suit is not None:
            return GameContract(
                spielart="Wenz",
                trumpf_cards=UNTER | SUIT_MASKS[suit],
                caller_id=caller_id,
            )
        return GameContract(spielart="Wenz", trumpf_cards=UNTER, caller_id=caller_id)

    @staticmethod
    def create_geier(caller_id: int) -> GameContract:
        return GameContract(spielart="Geier", trumpf_cards=OBER, caller_id=caller_id)

    @staticmethod
    def create_solo(caller_id: int, suit: Suit = Suit.Herz) -> GameContract:
        return GameContract(
            spielart="Solo",
            trumpf_cards=OBER | UNTER | SUIT_MASKS[suit],
            caller_id=caller_id,
        )

    @staticmethod
    def create_ramsch() -> GameContract:
        return GameContract(spielart="Ramsch", trumpf_cards=OBER | UNTER | HERZ)
