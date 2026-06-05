"""
Tests for get_legal_moves in non-Sauspiel contracts (Wenz, Solo, Geier).
Covers: trumpf follower logic, Farbe follower logic, free player, lead player.
"""

import pytest

from omados.engine.cards import CARD_TO_INDEX, Cards, Suit
from omados.engine.modes import GameModeFactory
from omados.engine.rules import get_legal_moves

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wenz():
    return GameModeFactory.create_wenz(caller_id=0)


@pytest.fixture
def herz_solo():
    return GameModeFactory.create_solo(caller_id=0, suit=Suit.Herz)


@pytest.fixture
def geier():
    return GameModeFactory.create_geier(caller_id=0)


# ---------------------------------------------------------------------------
# Lead player — always gets full hand back regardless of contract
# ---------------------------------------------------------------------------


def test_lead_returns_full_hand_wenz(wenz):
    hand = Cards(["EU", "GU", "EA", "E10", "EK", "G7", "S8", "H9"])
    legal = get_legal_moves(hand, lead_card_idx=None, contract=wenz)
    assert legal.sum() == hand.sum()
    assert legal.to_list() == hand.to_list()


def test_lead_returns_full_hand_solo(herz_solo):
    hand = Cards(["EO", "HU", "HA", "H10", "HK", "E7", "G8", "S9"])
    legal = get_legal_moves(hand, lead_card_idx=None, contract=herz_solo)
    assert legal.sum() == hand.sum()


def test_lead_returns_full_hand_geier(geier):
    hand = Cards(["EO", "GO", "EA", "GA", "E7", "G7", "S7", "H9"])
    legal = get_legal_moves(hand, lead_card_idx=None, contract=geier)
    assert legal.sum() == hand.sum()


# ---------------------------------------------------------------------------
# Wenz — only Unters are trump
# ---------------------------------------------------------------------------


def test_wenz_follower_must_follow_unter_trumpf(wenz):
    """When an Unter (trump) is led, follower must play their Unters."""
    # Hand has EU (trump) and lots of other cards
    hand = Cards(["EU", "EA", "E10", "EK", "G7", "G8", "S7", "S8"])
    eu_idx = CARD_TO_INDEX["EU"]  # Lead: Eichel Unter (trump)
    legal = get_legal_moves(hand, lead_card_idx=eu_idx, contract=wenz)

    # Only EU from hand is trump — must play it
    assert legal.to_list() == ["EU"]


def test_wenz_follower_no_trump_must_follow_farbe(wenz):
    """
    When an Unter is led but follower has no Unters, they follow Farbe or play free.
    """
    # No Unters in hand. Lead: EU (trump). Must play freely.
    hand = Cards(["EO", "EA", "E10", "EK", "G7", "G8", "S7", "S8"])
    eu_idx = CARD_TO_INDEX["EU"]
    legal = get_legal_moves(hand, lead_card_idx=eu_idx, contract=wenz)

    # No trump → can play anything
    assert legal.sum() == hand.sum()


def test_wenz_ober_is_farbe_not_trump(wenz):
    """In Wenz, leading an Ober means followers play that suit (Obers are Farbe)."""
    # Lead: EO (Eichel Ober — NOT trump in Wenz, it's just Eichel Farbe)
    # Hand has Eichel cards (non-Unter) and other suits
    hand = Cards(["EA", "E10", "EK", "E9", "G7", "G8", "S7", "S8"])
    eo_idx = CARD_TO_INDEX["EO"]
    legal = get_legal_moves(hand, lead_card_idx=eo_idx, contract=wenz)

    # Must follow Eichel Farbe (EA, E10, EK, E9 are all Eichel and non-trump in Wenz)
    eichel_farbe_in_hand = ["EA", "E10", "EK", "E9"]
    for card in eichel_farbe_in_hand:
        assert card in legal.to_list(), f"{card} should be playable"
    assert "G7" not in legal.to_list()
    assert "S7" not in legal.to_list()


def test_wenz_follower_free_when_no_matching_farbe(wenz):
    """Follower has neither trump nor matching Farbe → can play anything."""
    # Lead: GO (Gras Ober — Farbe Gras in Wenz). Hand has no Gras non-trump.
    hand = Cards(["EU", "EA", "E10", "EK", "H9", "H8", "S7", "S8"])
    go_idx = CARD_TO_INDEX["GO"]
    legal = get_legal_moves(hand, lead_card_idx=go_idx, contract=wenz)

    assert legal.sum() == hand.sum()


# ---------------------------------------------------------------------------
# Solo (Herz) — Obers + Unters + all Herz cards are trump
# ---------------------------------------------------------------------------


def test_solo_follower_must_follow_trumpf(herz_solo):
    """When a trump is led in Solo, follower must play trump if they have one."""
    # HO is lead (trump). Hand has H7 (trump) and Eichel cards.
    hand = Cards(["H7", "EA", "E10", "EK", "G7", "G8", "S7", "S8"])
    ho_idx = CARD_TO_INDEX["HO"]
    legal = get_legal_moves(hand, lead_card_idx=ho_idx, contract=herz_solo)

    # H7 is the only trump in hand
    assert legal.to_list() == ["H7"]


def test_solo_follower_must_follow_farbe(herz_solo):
    """When a non-trump suit is led, follower must play that suit (non-trump)."""
    # Lead: EA (Eichel Ace — Farbe, not trump in Herz Solo)
    # Hand has Eichel Farbe and other things
    hand = Cards(["E10", "EK", "E9", "G7", "G8", "S7", "S8", "H7"])
    ea_idx = CARD_TO_INDEX["EA"]
    legal = get_legal_moves(hand, lead_card_idx=ea_idx, contract=herz_solo)

    # Must follow Eichel Farbe (non-trump): E10, EK, E9
    assert "E10" in legal.to_list()
    assert "EK" in legal.to_list()
    assert "E9" in legal.to_list()
    assert "G7" not in legal.to_list()
    assert "H7" not in legal.to_list()  # H7 is trump, not Eichel Farbe


def test_solo_obers_and_unters_count_as_trump_not_farbe(herz_solo):
    """In Solo, EO is trump. Leading EA (Eichel Farbe) must NOT be satisfied by EO."""
    # Lead: EA. Hand has EO (trump, counts as trump not Eichel Farbe) and Gras.
    hand = Cards(["EO", "G7", "G8", "S7", "S8", "H9", "H8", "H7"])
    ea_idx = CARD_TO_INDEX["EA"]
    legal = get_legal_moves(hand, lead_card_idx=ea_idx, contract=herz_solo)

    # EO is trump (not Eichel Farbe) — no Eichel Farbe in hand → play freely
    assert legal.sum() == hand.sum()


def test_solo_follower_free_no_suit_no_trump(herz_solo):
    """No matching Farbe and no trump → can play anything."""
    # Lead: EA (Eichel). Hand has only Gras and Schellen (non-trump, non-Eichel).
    hand = Cards(["G7", "G8", "G9", "GK", "S7", "S8", "S9", "SK"])
    ea_idx = CARD_TO_INDEX["EA"]
    legal = get_legal_moves(hand, lead_card_idx=ea_idx, contract=herz_solo)

    assert legal.sum() == hand.sum()


# ---------------------------------------------------------------------------
# Geier — only Obers are trump
# ---------------------------------------------------------------------------


def test_geier_follower_must_follow_ober_trumpf(geier):
    """In Geier, when an Ober (trump) is led, follower plays their Obers."""
    hand = Cards(["GO", "EU", "GU", "EA", "G7", "S7", "H7", "S8"])
    eo_idx = CARD_TO_INDEX["EO"]
    legal = get_legal_moves(hand, lead_card_idx=eo_idx, contract=geier)

    # GO is the only Ober (trump) in hand
    assert legal.to_list() == ["GO"]


def test_geier_unter_is_farbe_not_trump(geier):
    """
    In Geier, leading an Unter means others follow that Farbe (Unter is not trump).
    """
    # Lead: EU (Eichel Unter — Farbe in Geier)
    hand = Cards(["GU", "EA", "E10", "EK", "G7", "G8", "S7", "S8"])
    eu_idx = CARD_TO_INDEX["EU"]
    legal = get_legal_moves(hand, lead_card_idx=eu_idx, contract=geier)

    # GU is Gras Unter (Farbe Gras in Geier, not Eichel)
    # No Eichel Farbe non-trump... wait, EU is Eichel. EA, E10, EK are Eichel Farbe.
    # GU is Gras Unter = Farbe Gras in Geier.
    # So follower has EA, E10, EK as Eichel Farbe — must play those.
    assert "EA" in legal.to_list()
    assert "E10" in legal.to_list()
    assert "EK" in legal.to_list()
    assert "GU" not in legal.to_list()  # Gras, not Eichel
    assert "G7" not in legal.to_list()


def test_geier_follower_free_no_trump_no_farbe(geier):
    """No Obers and no matching Farbe → play anything."""
    # Lead: EO (trump). Hand has no Obers.
    hand = Cards(["EA", "GA", "G7", "S7", "H7", "H8", "H9", "HK"])
    eo_idx = CARD_TO_INDEX["EO"]
    legal = get_legal_moves(hand, lead_card_idx=eo_idx, contract=geier)

    assert legal.sum() == hand.sum()


# ---------------------------------------------------------------------------
# One-card hand edge case
# ---------------------------------------------------------------------------


def test_single_card_hand_always_legal(wenz):
    """A player with one card can always play it, regardless of what was led."""
    hand = Cards(["G7"])
    eu_idx = CARD_TO_INDEX["EU"]  # Trump was led; G7 is not trump
    legal = get_legal_moves(hand, lead_card_idx=eu_idx, contract=wenz)

    assert legal.to_list() == ["G7"]
