"""
Tests for bidding eligibility functions in rules.py:
- can_play_sauspiel
- get_available_spielarten
- get_viable_spielarten
"""

from omados.agents.base import BiddingState
from omados.engine.cards import Cards, Suit
from omados.engine.modes import SPIELART_RANK, Spielart
from omados.engine.rules import (
    can_play_sauspiel,
    get_available_spielarten,
    get_viable_spielarten,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_bidding_state(
    player_id: int = 0,
    current_highest: Spielart | None = None,
    is_challenger: bool = False,
    current_right_holder: int = -1,
) -> BiddingState:
    return BiddingState(
        hand=Cards([]),
        player_id=player_id,
        dealer_id=0,
        declarations={},
        bid_history=[],
        current_highest_spielart=current_highest,
        current_right_holder=current_right_holder,
        is_challenger=is_challenger,
    )


# ---------------------------------------------------------------------------
# can_play_sauspiel
# ---------------------------------------------------------------------------


def test_can_play_sauspiel_standard_eligible():
    """Has Eichel Farbe, no Eichel Ace → Eichel is callable."""
    hand = Cards(["EO", "GO", "EU", "HU", "E7", "E8", "G10", "GK"])
    suits = can_play_sauspiel(hand)
    assert Suit.Eichel in suits


def test_can_play_sauspiel_blocked_by_ace():
    """Holds EA → cannot call Eichel."""
    hand = Cards(["EO", "GO", "EU", "HU", "EA", "E7", "G7", "G8"])
    suits = can_play_sauspiel(hand)
    assert Suit.Eichel not in suits
    assert Suit.Gras in suits


def test_can_play_sauspiel_no_farbe_cards():
    """Has only trumpf (Obers/Unters/Herz) — no non-trump Farbe → empty."""
    hand = Cards(["EO", "GO", "HO", "SO", "EU", "GU", "HU", "SU"])
    suits = can_play_sauspiel(hand)
    assert suits == []


def test_can_play_sauspiel_ace_only_no_farbe():
    """Has SA but no other Schellen → cannot call Schellen (ace-only)."""
    hand = Cards(["EO", "GO", "HO", "SO", "EU", "GU", "HU", "SA"])
    suits = can_play_sauspiel(hand)
    assert Suit.Schellen not in suits


def test_can_play_sauspiel_herz_never_callable():
    """Herz is always trump in Sauspiel — never a callable Rufsau suit."""
    hand = Cards(["H10", "HK", "H9", "H8", "H7", "EO", "GO", "E7"])
    suits = can_play_sauspiel(hand)
    assert Suit.Herz not in suits


def test_can_play_sauspiel_multiple_suits():
    """Can call multiple suits at once."""
    hand = Cards(["EO", "EU", "E7", "E8", "G7", "G8", "S7", "S8"])
    suits = can_play_sauspiel(hand)
    assert Suit.Eichel in suits
    assert Suit.Gras in suits
    assert Suit.Schellen in suits


def test_can_play_sauspiel_ober_unter_dont_count_as_farbe():
    """EO/EU are trump — don't count as Eichel Farbe for Sauspiel eligibility."""
    # Only Eichel cards in hand are EO (Ober) and EU (Unter) — both trump
    hand = Cards(["EO", "EU", "HA", "H10", "HK", "GA", "G10", "GK"])
    suits = can_play_sauspiel(hand)
    assert Suit.Eichel not in suits  # No non-trump Eichel Farbe


# ---------------------------------------------------------------------------
# get_available_spielarten
# ---------------------------------------------------------------------------


def test_available_spielarten_always_has_wenz_geier_solo():
    """Every hand can play Wenz, Geier, and Solo regardless of cards."""
    hand = Cards(["EO", "GO", "HO", "SO", "EU", "GU", "HU", "SU"])
    spielarten = get_available_spielarten(hand)
    assert Spielart.Wenz in spielarten
    assert Spielart.Geier in spielarten
    assert Spielart.Solo in spielarten


def test_available_spielarten_no_duplicates():
    """Each Spielart appears at most once."""
    hand = Cards(["EO", "EU", "E7", "E8", "G7", "G8", "S7", "S8"])
    spielarten = get_available_spielarten(hand)
    assert len(spielarten) == len(set(spielarten))


def test_available_spielarten_no_sauspiel_when_all_aces():
    """Cannot call Sauspiel if holding all non-Herz aces."""
    hand = Cards(["EO", "GO", "EU", "GU", "EA", "GA", "SA", "HU"])
    spielarten = get_available_spielarten(hand)
    assert Spielart.Sauspiel not in spielarten


def test_available_spielarten_has_sauspiel_when_eligible():
    hand = Cards(["EO", "EU", "E7", "E8", "G7", "G8", "S7", "S8"])
    spielarten = get_available_spielarten(hand)
    assert Spielart.Sauspiel in spielarten


def test_available_spielarten_includes_farbwenz_farbgeier():
    hand = Cards(["EO", "EU", "E7", "E8", "G7", "G8", "S7", "S8"])
    spielarten = get_available_spielarten(hand)
    assert Spielart.Farbwenz in spielarten
    assert Spielart.Farbgeier in spielarten


def test_available_spielarten_ordered_by_rank():
    """Result should be in ascending rank order (low → high)."""
    hand = Cards(["EO", "EU", "E7", "E8", "G7", "G8", "S7", "H9"])
    spielarten = get_available_spielarten(hand)
    ranks = [SPIELART_RANK[s] for s in spielarten]
    assert ranks == sorted(ranks)


# ---------------------------------------------------------------------------
# get_viable_spielarten — bidding_position floor
# ---------------------------------------------------------------------------


def test_viable_no_current_highest_position_0():
    """Position 0: all available spielarten are viable."""
    hand = Cards(["EO", "EU", "E7", "E8", "G7", "G8", "S7", "H9"])
    state = make_bidding_state()
    viable = get_viable_spielarten(hand, state, bidding_position=0)
    available = get_available_spielarten(hand)
    assert set(viable) == set(available)


def test_viable_position_floor_filters_low_spielarten():
    """bidding_position=4 (Wenz rank) → only Wenz and above are viable."""
    hand = Cards(["EO", "EU", "E7", "E8", "G7", "G8", "S7", "H9"])
    state = make_bidding_state()
    viable = get_viable_spielarten(hand, state, bidding_position=4)

    for s in viable:
        assert SPIELART_RANK[s] >= 4, f"{s} should be filtered by position floor"

    # Sauspiel (rank 0), Farbgeier (1), Farbwenz (2), Geier (3) should all be excluded
    assert Spielart.Sauspiel not in viable
    assert Spielart.Geier not in viable


def test_viable_position_floor_allows_exact_rank():
    """Position=3 (Geier rank) → Geier itself must be included."""
    hand = Cards(["EO", "EU", "E7", "E8", "G7", "G8", "S7", "H9"])
    state = make_bidding_state()
    viable = get_viable_spielarten(hand, state, bidding_position=3)
    assert Spielart.Geier in viable


def test_viable_high_position_only_solo_and_above():
    """Position=5 (Solo rank) → only Solo (and Sie if available)."""
    hand = Cards(["EO", "EU", "E7", "E8", "G7", "G8", "S7", "H9"])
    state = make_bidding_state()
    viable = get_viable_spielarten(hand, state, bidding_position=5)
    for s in viable:
        assert SPIELART_RANK[s] >= 5
    assert Spielart.Solo in viable


# ---------------------------------------------------------------------------
# get_viable_spielarten — current_highest_spielart (challenger must beat it)
# ---------------------------------------------------------------------------


def test_viable_challenger_must_exceed_current():
    """Challenger must bid strictly higher than current highest."""
    hand = Cards(["EO", "EU", "E7", "E8", "G7", "G8", "S7", "H9"])
    state = make_bidding_state(
        current_highest=Spielart.Geier,  # rank 3
        is_challenger=True,
    )
    viable = get_viable_spielarten(hand, state, bidding_position=0)

    for s in viable:
        assert SPIELART_RANK[s] > SPIELART_RANK[Spielart.Geier], (
            f"{s} is not strictly higher than Geier"
        )
    assert Spielart.Geier not in viable
    assert Spielart.Sauspiel not in viable


def test_viable_challenger_current_highest_wenz_must_play_solo_or_higher():
    hand = Cards(["EO", "EU", "E7", "E8", "G7", "G8", "S7", "H9"])
    state = make_bidding_state(current_highest=Spielart.Wenz, is_challenger=True)
    viable = get_viable_spielarten(hand, state, bidding_position=0)

    for s in viable:
        assert SPIELART_RANK[s] > SPIELART_RANK[Spielart.Wenz]
    assert Spielart.Solo in viable


# ---------------------------------------------------------------------------
# get_viable_spielarten — defender can match current
# ---------------------------------------------------------------------------


def test_viable_defender_can_match_current_highest():
    """Defender can play at least the current highest or anything above."""
    hand = Cards(["EO", "EU", "E7", "E8", "G7", "G8", "S7", "H9"])
    state = make_bidding_state(
        current_highest=Spielart.Geier,  # rank 3
        is_challenger=False,
    )
    viable = get_viable_spielarten(hand, state, bidding_position=0)

    assert Spielart.Geier in viable  # can match
    assert Spielart.Wenz in viable  # can exceed
    for s in viable:
        assert SPIELART_RANK[s] >= SPIELART_RANK[Spielart.Geier]


def test_viable_defender_cannot_play_below_current():
    hand = Cards(["EO", "EU", "E7", "E8", "G7", "G8", "S7", "H9"])
    state = make_bidding_state(current_highest=Spielart.Wenz, is_challenger=False)
    viable = get_viable_spielarten(hand, state, bidding_position=0)

    assert Spielart.Sauspiel not in viable
    assert Spielart.Geier not in viable


# ---------------------------------------------------------------------------
# get_viable_spielarten — position floor + current_highest combined
# ---------------------------------------------------------------------------


def test_viable_position_and_current_highest_both_apply():
    """Both filters apply simultaneously — take the stricter one."""
    hand = Cards(["EO", "EU", "E7", "E8", "G7", "G8", "S7", "H9"])
    # Position=2 (Farbwenz floor), current_highest=Geier (rank 3), challenger
    state = make_bidding_state(current_highest=Spielart.Geier, is_challenger=True)
    viable = get_viable_spielarten(hand, state, bidding_position=2)

    # Must be >= position floor (2) AND > current_highest rank (3) → effectively > 3
    for s in viable:
        assert SPIELART_RANK[s] > 3
    assert Spielart.Farbwenz not in viable  # rank 2 < 3
    assert Spielart.Geier not in viable  # not strictly higher (challenger)
    assert Spielart.Wenz in viable  # rank 4 > 3 ✓
    assert Spielart.Solo in viable  # rank 5 > 3 ✓


def test_viable_empty_when_position_too_high_for_hand():
    """If position floor exceeds all available spielarten, result is empty."""
    # Hand with no special cards — can play up to Solo (rank 5) max
    hand = Cards(["EO", "EU", "E7", "E8", "G7", "G8", "S7", "H9"])
    state = make_bidding_state(current_highest=Spielart.Solo, is_challenger=True)
    # Challenger must beat Solo → only Sie (rank 6), but hand can't play Sie
    viable = get_viable_spielarten(hand, state, bidding_position=0)
    assert Spielart.Solo not in viable
    # No Sie in hand → empty (or just Sie if present, which it isn't here)
    for s in viable:
        assert SPIELART_RANK[s] > SPIELART_RANK[Spielart.Solo]
