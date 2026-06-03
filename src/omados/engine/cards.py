import logging
from collections.abc import Callable
from enum import Enum
from functools import partial

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# 1. Use Enums for Suits (Farben) and Ranks (Values)
class Suit(str, Enum):
    Eichel = "E"
    Gras = "G"
    Herz = "H"
    Schellen = "S"


class Rank(str, Enum):
    Ober = "O"
    Unter = "U"
    Ace = "A"
    Ten = "10"
    King = "K"
    Nine = "9"
    Eight = "8"
    Seven = "7"

    @property
    def points(self) -> int:
        # Maps the rank to its point value
        # Note: I kept 7 as 8 points based on your original code,
        # though in standard Schafkopf it is typically 0.
        point_mapping = {
            "O": 3,
            "U": 2,
            "A": 11,
            "10": 10,
            "K": 4,
            "9": 0,
            "8": 0,
            "7": 0,
        }
        return point_mapping[self.value]


# 2. Generate the deck dynamically to preserve your specific order
# (Obers and Unters first, then remaining cards sorted by Suit)
_OBERS = [f"{suit.value}{Rank.Ober.value}" for suit in Suit]
_UNTERS = [f"{suit.value}{Rank.Unter.value}" for suit in Suit]
_REST = [
    f"{suit.value}{rank.value}"
    for suit in [Suit.Herz, Suit.Eichel, Suit.Gras, Suit.Schellen]
    for rank in Rank
    if rank not in (Rank.Ober, Rank.Unter)
]

DECK = _OBERS + _UNTERS + _REST
CARD_TO_INDEX = {card: index for index, card in enumerate(DECK)}

# 3. Create the points tensor cleanly
CARD_POINTS = torch.tensor(
    [
        Rank(card[1:]).points if not card.startswith("10") else Rank.Ten.points
        for card in DECK
    ]
)

# Precompute a tuple where the index matches the card index in DECK
INDEX_TO_SUIT = tuple(Suit(card[0]) for card in DECK)


class Cards:
    def __init__(self, data: list[str] | torch.Tensor):
        """Initialize with either a list of card strings or a boolean PyTorch tensor."""
        if isinstance(data, list):
            self.cards = self._list_to_tensor(data)
        elif isinstance(data, torch.Tensor):
            self.cards = data
        else:
            raise TypeError("Data must be a list of cards or a boolean torch.Tensor.")

    @staticmethod
    def _list_to_tensor(cards_list: list[str]) -> torch.Tensor:
        tensor = torch.zeros(32, dtype=torch.bool)
        indices = [CARD_TO_INDEX[card] for card in cards_list]
        tensor[indices] = True
        return tensor

    def to_list(self) -> list[str]:
        """Converts the internal tensor back to a list of card strings."""
        indices = torch.where(self.cards)[0].tolist()
        return [DECK[i] for i in indices]

    # Operator overloading
    def __and__(self, other: "Cards") -> "Cards":
        if not isinstance(other, Cards):
            return NotImplemented
        return Cards(self.cards & other.cards)

    def __or__(self, other: "Cards") -> "Cards":
        if not isinstance(other, Cards):
            return NotImplemented
        return Cards(self.cards | other.cards)

    def __invert__(self) -> "Cards":
        return Cards(~self.cards)

    def __repr__(self) -> str:
        return f"Cards({self.to_list()})"

    def __getitem__(self, key: int) -> torch.Tensor:
        return self.cards[key]

    def any(self) -> bool:
        return any(self.cards)

    def sum(self) -> Tensor:
        return self.cards.sum()


# 4. Generate Precomputed Masks using dictionary comprehensions
def _create_mask(condition_func: Callable[[str], bool]) -> Cards:
    tensor = torch.zeros(32, dtype=torch.bool)
    for i, card in enumerate(DECK):
        if condition_func(card):
            tensor[i] = True
    return Cards(tensor)


def _suit_condition(c: str, s: Suit) -> bool:
    return c.startswith(s.value)


def _rank_condition(c: str, r: Rank) -> bool:
    return c.endswith(r.value)


# Masks mapped directly to Enums for clean importing and usage
SUIT_MASKS = {suit: _create_mask(partial(_suit_condition, s=suit)) for suit in Suit}
RANK_MASKS = {rank: _create_mask(partial(_rank_condition, r=rank)) for rank in Rank}

SUIT_MATRIX = torch.stack([SUIT_MASKS[s].cards for s in Suit])
SUIT_LOOKUP = torch.argmax(SUIT_MATRIX.to(torch.int8), dim=0).to(
    torch.int64
)  # takes a bit too much space but since torch uses int64 internally, we just use those

# 5. Export explicit constants for clean module-level imports and easy bitwise logic
SCHELLEN = SUIT_MASKS[Suit.Schellen]
GRAS = SUIT_MASKS[Suit.Gras]
EICHEL = SUIT_MASKS[Suit.Eichel]
HERZ = SUIT_MASKS[Suit.Herz]

OBER = RANK_MASKS[Rank.Ober]
UNTER = RANK_MASKS[Rank.Unter]
ASSE = RANK_MASKS[Rank.Ace]
ZEHNER = RANK_MASKS[Rank.Ten]
KOENIGE = RANK_MASKS[Rank.King]
NEUNER = RANK_MASKS[Rank.Nine]
ACHTER = RANK_MASKS[Rank.Eight]
SIEBENER = RANK_MASKS[Rank.Seven]

NO_CARDS = Cards([])
