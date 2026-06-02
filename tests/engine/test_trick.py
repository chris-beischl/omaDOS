import pytest

from omados.engine.cards import CARD_TO_INDEX, Cards
from omados.engine.modes import GameModeFactory
from omados.engine.tricks import Trick


def test_trick_initialization_empty():
    """Verify an empty trick sets up the player order correctly."""
    lead_id = 2
    trick = Trick(lead_player_id=lead_id)

    # Expected clockwise: 2 -> 3 -> 0 -> 1
    assert trick.player_ids == [2, 3, 0, 1]
    assert trick.is_full is False
    assert len(trick.card_indices) == 0
    # Check that the players_tensor is pre-filled correctly
    assert trick.players_tensor[0, 2] == 1.0  # First turn is player 2
    assert trick.players_tensor[1, 3] == 1.0  # Second turn is player 3


def test_trick_add_card_sequential():
    """Verify that adding cards updates the state and tensors correctly."""
    trick = Trick(lead_player_id=0)

    # Add first card as string
    trick.add_card("EO")
    assert trick.card_indices == [CARD_TO_INDEX["EO"]]
    assert trick.cards_tensor[0, CARD_TO_INDEX["EO"]] == 1.0

    # Add second card as index
    g7_idx = CARD_TO_INDEX["G7"]
    trick.add_card(g7_idx)
    assert len(trick.card_indices) == 2
    assert trick.cards_tensor[1, g7_idx] == 1.0
    assert trick.is_full is False


def test_trick_add_cards_object():
    """Verify that a single-card Cards object can be added."""
    trick = Trick(lead_player_id=0)
    single_card = Cards(["HA"])

    trick.add_card(single_card)
    assert trick.card_indices[0] == CARD_TO_INDEX["HA"]


def test_trick_full_validation():
    """Verify the trick identifies as full and prevents over-filling."""
    trick = Trick(lead_player_id=0, cards=["EO", "GO", "HO", "SO"])

    assert trick.is_full is True
    with pytest.raises(ValueError, match="already full"):
        trick.add_card("SA")


def test_trick_points(sauspiel_eichel):
    """Verify points calculation (Augen) within a trick."""
    # EO(3), G7(0), HA(11), S10(10) = 24
    cards = ["EO", "G7", "HA", "S10"]
    trick = Trick(lead_player_id=0, cards=cards)

    assert trick.points == 24


def test_trick_contains_trumpf(sauspiel_eichel):
    """Verify trump detection logic across different contracts."""
    # 1. Sauspiel (Herz is Trump)
    trick_with_trump = Trick(lead_player_id=0, cards=["E7", "H7", "G8", "S9"])
    assert trick_with_trump.contains_trumpf(sauspiel_eichel) is True

    trick_no_trump = Trick(lead_player_id=0, cards=["E7", "G7", "S7", "S8"])
    assert trick_no_trump.contains_trumpf(sauspiel_eichel) is False

    # 2. Wenz (Only Unters are Trump)
    wenz = GameModeFactory.create_wenz(caller_id=0)
    # In Wenz, Obers are NOT trump
    trick_wenz_ober = Trick(lead_player_id=0, cards=["EO", "GO", "E7", "E8"])
    assert trick_wenz_ober.contains_trumpf(wenz) is False

    trick_wenz_unter = Trick(lead_player_id=0, cards=["EU"])
    assert trick_wenz_unter.contains_trumpf(wenz) is True


def test_trick_repr():
    """Visual check for the debugging string."""
    trick = Trick(lead_player_id=1, cards=["EO", "G7"])
    r = repr(trick)
    assert "lead=1" in r
    assert "'EO', 'G7'" in r
    assert "played_by=[1, 2]" in r
