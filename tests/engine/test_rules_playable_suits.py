import pytest

from omados.engine.cards import HERZ, OBER, UNTER, Cards, Suit
from omados.engine.rules import get_playable_suits


@pytest.fixture
def sauspiel_trumpf():
    """Standard Sauspiel Trumpf: All Obers, all Unters, and all Hearts."""
    return OBER | UNTER | HERZ


def test_sauspiel_standard_eligibility(sauspiel_trumpf):
    """
    Hand: 4 Trumps, 2 Eichels (7, 8), 2 Gras (10, K).
    Result: Can call Eichel and Gras (has cards, no Aces).
    """
    # 4 Trumps + 2 Eichel + 2 Gras = 8 cards
    hand = Cards(["EO", "GO", "EU", "HU", "E7", "E8", "G10", "GK"])
    suits = get_playable_suits(hand, sauspiel_trumpf)

    assert Suit.Eichel in suits
    assert Suit.Gras in suits
    assert Suit.Schellen not in suits  # No Schellen cards held
    assert Suit.Herz not in suits  # Herz is Trump, never a 'suit' in Sauspiel


def test_sauspiel_blocked_by_owning_the_ace(sauspiel_trumpf):
    """
    Hand: 4 Trumps, 2 Eichels (Ace, 7), 2 Gras (7, 8).
    Result: Only Gras is callable. Eichel is blocked by EA.
    """
    hand = Cards(["EO", "GO", "EU", "HU", "EA", "E7", "G7", "G8"])
    suits = get_playable_suits(hand, sauspiel_trumpf)

    assert Suit.Eichel not in suits  # Blocked: player has the Ace
    assert Suit.Gras in suits  # Valid: has cards, no Ace


def test_sauspiel_besetzt_ace_only(sauspiel_trumpf):
    """
    Hand: 7 Trumps, 1 Schellen Ace.
    Result: Empty list. You cannot call Schellen if you only have the Ace.
    """
    # 7 Trumps + 1 Ace = 8 cards
    hand = Cards(["EO", "GO", "HO", "SO", "EU", "GU", "HU", "SA"])
    suits = get_playable_suits(hand, sauspiel_trumpf)

    assert suits == []


def test_sauspiel_no_suit_cards(sauspiel_trumpf):
    """
    Hand: 8 Trumps.
    Result: Empty list. Cannot play Sauspiel if you have no non-trump cards.
    """
    hand = Cards(["EO", "GO", "HO", "SO", "EU", "GU", "HU", "SU"])
    suits = get_playable_suits(hand, sauspiel_trumpf)

    assert suits == []


def test_sauspiel_mixed_hand_all_aces(sauspiel_trumpf):
    """
    Hand: 5 Trumps, 3 Aces (EA, GA, SA).
    Result: Empty list. Player owns all the Aces they could potentially call.
    """
    hand = Cards(["EO", "GO", "HO", "EU", "GU", "EA", "GA", "SA"])
    suits = get_playable_suits(hand, sauspiel_trumpf)

    assert suits == []


def test_sauspiel_many_cards_no_ace(sauspiel_trumpf):
    """
    Result: Schellen is callable.
    """
    # Let's fix that hand to be exactly 8:
    # 3 Trumps (EO, GO, HO) + 5 Schellen (10, K, 9, 8, 7)
    hand = Cards(["EO", "GO", "HO", "S10", "SK", "S9", "S8", "S7"])
    suits = get_playable_suits(hand, sauspiel_trumpf)

    assert suits == [Suit.Schellen]


def test_herz_is_never_playable_full_heart_hand(sauspiel_trumpf):
    """
    Even if a player has every single Heart card (Ace through 7),
    none of them are 'suit' cards in Sauspiel.
    """
    # 6 Hearts (non-Ober/Unter) + 2 Obers = 8 cards
    hand = Cards(["HA", "H10", "HK", "H9", "H8", "H7", "EO", "GO"])
    suits = get_playable_suits(hand, sauspiel_trumpf)

    # Logic: non_trumpf_cards will be empty because all 8 cards are in sauspiel_trumpf.
    assert Suit.Herz not in suits
    assert suits == []


def test_herz_is_never_playable_missing_ace(sauspiel_trumpf):
    """
    Even if you have 'small' Hearts (H10, HK, etc.) and NO Heart Ace,
    they are still Trump and thus not a 'callable' suit.
    """
    # 5 Hearts (no Ace) + 2 Obers + 1 Eichel 7
    hand = Cards(["H10", "HK", "H9", "H8", "H7", "EO", "GO", "E7"])
    suits = get_playable_suits(hand, sauspiel_trumpf)

    # Eichel 7 is the only non-trump card.
    # It meets the criteria: Have E7, don't have EA.
    assert Suit.Herz not in suits
    assert Suit.Eichel in suits


def test_herz_ace_as_only_heart(sauspiel_trumpf):
    """
    A hand with the Herz Ace and 7 other trumps.
    """
    # Herz Ace + 7 other trumps (Obers/Unters)
    hand = Cards(["HA", "EO", "GO", "HO", "SO", "EU", "GU", "HU"])
    suits = get_playable_suits(hand, sauspiel_trumpf)

    assert suits == []


def test_herz_ignored_with_other_callable_suits(sauspiel_trumpf):
    """
    Hand: 4 Hearts, 2 Eichels (no Ace), 2 Gras (no Ace).
    Verify Herz is filtered out while others remain.
    """
    # 4 Hearts + 2 Eichel (E7, E8) + 2 Gras (G7, G8)
    hand = Cards(["H10", "HK", "H9", "H8", "E7", "E8", "G7", "G8"])
    suits = get_playable_suits(hand, sauspiel_trumpf)

    assert Suit.Herz not in suits
    assert Suit.Eichel in suits
    assert Suit.Gras in suits
    assert len(suits) == 2
