import pytest
import torch

from omados.engine.cards import (
    ASSE,
    CARD_POINTS,
    CARD_TO_INDEX,
    DECK,
    EICHEL,
    INDEX_TO_SUIT,
    OBER,
    SUIT_LOOKUP,
    Cards,
    Suit,
)

# --- 1. Testing Deck & Constants ---


def test_deck_integrity():
    """Verify the deck has exactly 32 unique cards and follows the specific order."""
    assert len(DECK) == 32
    assert len(set(DECK)) == 32
    # Obers should be the first 4
    for i in range(4):
        assert DECK[i].endswith("O")
    # Unters should be the next 4
    for i in range(4, 8):
        assert DECK[i].endswith("U")


def test_card_indices():
    """Verify mapping from string to index and back."""
    assert CARD_TO_INDEX["EO"] == 0
    assert DECK[CARD_TO_INDEX["SA"]] == "SA"


def test_points():
    """Verify that points are correctly assigned to each card."""
    # Eichel Ober = 3 points
    assert CARD_POINTS[CARD_TO_INDEX["EO"]] == 3
    # Herz Zehn = 10 points
    assert CARD_POINTS[CARD_TO_INDEX["H10"]] == 10
    # Schellen Sieben = 0 points
    assert CARD_POINTS[CARD_TO_INDEX["S7"]] == 0
    # Total points in a Schafkopf deck must be 120
    assert CARD_POINTS.sum() == 120


def test_index_to_suit():
    """Verify that indices correctly map back to the Suit Enum."""
    assert INDEX_TO_SUIT[CARD_TO_INDEX["EO"]] == Suit.Eichel
    assert INDEX_TO_SUIT[CARD_TO_INDEX["S7"]] == Suit.Schellen


# --- 2. Testing the Cards Class ---


def test_cards_initialization_list():
    """Test creating Cards from a list of strings."""
    c = Cards(["EO", "GU", "HA"])
    assert c.sum() == 3
    assert c.to_list() == ["EO", "GU", "HA"]


def test_cards_initialization_tensor():
    """Test creating Cards from a pre-existing boolean tensor."""
    tensor = torch.zeros(32, dtype=torch.bool)
    tensor[0] = True  # EO
    c = Cards(tensor)
    assert "EO" in c.to_list()


def test_cards_invalid_init():
    """Verify that invalid types raise a TypeError."""
    with pytest.raises(TypeError):
        Cards(123)  # type: ignore


def test_cards_getitem():
    """Verify we can check a specific card index."""
    c = Cards(["EO"])
    assert c[0]
    assert not c[1]


# --- 3. Testing Logic & Operators ---


def test_bitwise_and():
    """Verify intersection logic."""
    c1 = Cards(["EO", "GO"])
    c2 = Cards(["GO", "HO"])
    result = c1 & c2
    assert result.to_list() == ["GO"]


def test_bitwise_or():
    """Verify union logic."""
    c1 = Cards(["EO"])
    c2 = Cards(["GO"])
    result = c1 | c2
    assert "EO" in result.to_list()
    assert "GO" in result.to_list()


def test_bitwise_invert():
    """Verify negation (e.g., getting all non-trump cards)."""
    # If we have only EO, the inverse should have 31 cards
    c = Cards(["EO"])
    inverted = ~c
    assert inverted.sum() == 31
    assert "EO" not in inverted.to_list()


# --- 4. Testing Precomputed Masks ---


def test_rank_masks():
    """Verify OBER, UNTER, etc. masks contain the correct cards."""
    assert OBER.sum() == 4
    assert OBER.to_list() == ["EO", "GO", "HO", "SO"]
    assert ASSE.sum() == 4
    assert "SA" in ASSE.to_list()


def test_suit_masks():
    """Verify EICHEL, HERZ, etc. masks contain 8 cards (including O/U)."""
    assert EICHEL.sum() == 8
    # Check if Eichel Ober is in Eichel mask
    assert "EO" in EICHEL.to_list()


def test_suit_lookup_matrix():
    """Verify the SUIT_LOOKUP tensor correctly identifies suits for indices."""
    eo_idx = CARD_TO_INDEX["EO"]
    # We expect the integer value of Suit.Eichel
    # Since Suit is a string Enum here, we check the SUIT_LOOKUP result matches the
    # Enum index
    expected_suit_idx = list(Suit).index(Suit.Eichel)
    assert SUIT_LOOKUP[eo_idx].item() == expected_suit_idx
