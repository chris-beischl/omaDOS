"""
Tests for Trick.lead_card property.
Covers: empty trick, single card, partial trick, full trick.
"""

from omados.engine.cards import CARD_TO_INDEX
from omados.engine.tricks import Trick


def test_lead_card_empty_trick_is_none():
    """An empty trick has no lead card."""
    trick = Trick(lead_player_id=0)
    assert trick.lead_card is None


def test_lead_card_after_first_card():
    """lead_card returns the index of the first card played."""
    trick = Trick(lead_player_id=1)
    trick.add_card("EO")
    assert trick.lead_card == CARD_TO_INDEX["EO"]


def test_lead_card_after_second_card_unchanged():
    """Adding a second card does not change lead_card."""
    trick = Trick(lead_player_id=2)
    trick.add_card("EO")
    trick.add_card("G7")
    assert trick.lead_card == CARD_TO_INDEX["EO"]


def test_lead_card_full_trick_still_first():
    """lead_card stays as the first card even when the trick is full."""
    trick = Trick(lead_player_id=0, cards=["HA", "G7", "S8", "E9"])
    assert trick.lead_card == CARD_TO_INDEX["HA"]
    assert trick.is_full


def test_lead_card_is_int_not_none_after_play():
    """Type is int once any card has been played."""
    trick = Trick(lead_player_id=3)
    assert trick.lead_card is None
    trick.add_card("SO")
    assert isinstance(trick.lead_card, int)


def test_lead_card_different_lead_players():
    """lead_card is purely about the card index, not the player — verify with
    various lead players."""
    for lead_pid in range(4):
        trick = Trick(lead_player_id=lead_pid)
        trick.add_card("HU")
        assert trick.lead_card == CARD_TO_INDEX["HU"]


def test_lead_card_constructed_with_cards_list():
    """Trick initialized with a cards list still exposes correct lead_card."""
    trick = Trick(lead_player_id=0, cards=["GK", "EA", "S10", "HO"])
    assert trick.lead_card == CARD_TO_INDEX["GK"]
