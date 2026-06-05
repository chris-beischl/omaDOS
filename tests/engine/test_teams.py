"""
Tests for get_teams in rules.py.
Covers: Sauspiel team detection across all Rufsau holder positions,
Solo/Wenz/Geier caller-alone setup, and Ramsch guard.
"""

import pytest

from omados.engine.cards import Cards, Suit
from omados.engine.modes import GameModeFactory
from omados.engine.rules import get_teams

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def empty_hand() -> Cards:
    return Cards([])


def hand_with(*cards: str) -> Cards:
    return Cards(list(cards))


# ---------------------------------------------------------------------------
# Sauspiel — rufsau holder joins the caller
# ---------------------------------------------------------------------------


def test_sauspiel_rufsau_holder_is_p1(rufsau_hand, sauspiel_eichel):
    """
    Caller is P0. P1 holds the EA (Eichel Ace).
    Expected teams: [0, 1] vs [2, 3].
    """
    hands = [
        hand_with("GO", "HO", "SO", "GU", "HU", "SU", "E10", "G10"),  # P0 caller
        # ["EO", "EU", "EA", "S10", "EK", "E9", "E8", "S7"]
        rufsau_hand,  # P1 holds EA
        hand_with("H7", "H8", "H9", "HK", "G7", "G8", "G9", "GK"),  # P2
        hand_with("S8", "S9", "SK", "E7", "SA", "GA", "H10", "HA"),  # P3
    ]

    player_team, opponent_team = get_teams(sauspiel_eichel, hands)

    assert sorted(player_team) == [0, 1]
    assert sorted(opponent_team) == [2, 3]


def test_sauspiel_rufsau_holder_is_p2():
    """Caller is P0. P2 holds the EA. Teams: [0,2] vs [1,3]."""
    contract = GameModeFactory.create_sauspiel(caller_id=0, suit=Suit.Eichel)
    hands = [
        hand_with("EO", "GO", "HO", "SO", "EU", "GU", "HU", "SU"),
        hand_with("H7", "H8", "H9", "HK", "G7", "G8", "G9", "GK"),
        hand_with("EA", "E10", "EK", "E9", "S7", "S8", "S9", "SK"),  # P2 holds EA
        hand_with("E7", "E8", "H10", "HA", "GA", "G10", "SA", "S10"),
    ]

    player_team, opponent_team = get_teams(contract, hands)

    assert sorted(player_team) == [0, 2]
    assert sorted(opponent_team) == [1, 3]


def test_sauspiel_rufsau_holder_is_p3():
    """Caller is P1. P3 holds the GA (Gras Ace). Teams: [1,3] vs [0,2]."""
    contract = GameModeFactory.create_sauspiel(caller_id=1, suit=Suit.Gras)
    hands = [
        hand_with("H7", "H8", "H9", "HK", "E7", "E8", "E9", "EK"),
        hand_with("EO", "GO", "HO", "SO", "EU", "GU", "HU", "SU"),  # P1 caller
        hand_with("EA", "E10", "S7", "S8", "S9", "SK", "SA", "S10"),
        hand_with("GA", "G10", "GK", "G9", "G8", "G7", "H10", "HA"),  # P3 holds GA
    ]

    player_team, opponent_team = get_teams(contract, hands)

    assert sorted(player_team) == [1, 3]
    assert sorted(opponent_team) == [0, 2]


def test_sauspiel_caller_holds_rufsau_himself():
    """
    Edge case: caller holds the Rufsau themselves.
    This shouldn't happen in a valid game (caller can't call a suit they have the Ace
    for), but get_teams should still find them and form team [caller, caller].
    We just verify it doesn't crash and caller appears in player_team.
    """
    contract = GameModeFactory.create_sauspiel(caller_id=0, suit=Suit.Eichel)
    hands = [
        hand_with("EA", "EO", "GO", "HO", "EU", "GU", "HU", "SU"),  # P0 holds EA
        hand_with("H7", "H8", "H9", "HK", "G7", "G8", "G9", "GK"),
        hand_with("E10", "EK", "E9", "S7", "S8", "S9", "SK", "SA"),
        hand_with("E7", "E8", "H10", "HA", "GA", "G10", "SO", "S10"),
    ]

    player_team, opponent_team = get_teams(contract, hands)
    assert 0 in player_team


# ---------------------------------------------------------------------------
# Solo — caller alone vs 3 opponents
# ---------------------------------------------------------------------------


def test_solo_caller_is_alone():
    """In Solo, only the caller is in player_team."""
    contract = GameModeFactory.create_solo(caller_id=2, suit=Suit.Herz)
    hands = [empty_hand(), empty_hand(), empty_hand(), empty_hand()]

    player_team, opponent_team = get_teams(contract, hands)

    assert player_team == [2]
    assert sorted(opponent_team) == [0, 1, 3]


def test_solo_caller_p0():
    contract = GameModeFactory.create_solo(caller_id=0, suit=Suit.Eichel)
    hands = [empty_hand()] * 4

    player_team, opponent_team = get_teams(contract, hands)

    assert player_team == [0]
    assert sorted(opponent_team) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Wenz — caller alone vs 3 opponents
# ---------------------------------------------------------------------------


def test_wenz_caller_is_alone():
    contract = GameModeFactory.create_wenz(caller_id=3)
    hands = [empty_hand()] * 4

    player_team, opponent_team = get_teams(contract, hands)

    assert player_team == [3]
    assert sorted(opponent_team) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Geier — caller alone vs 3 opponents
# ---------------------------------------------------------------------------


def test_geier_caller_is_alone():
    contract = GameModeFactory.create_geier(caller_id=1)
    hands = [empty_hand()] * 4

    player_team, opponent_team = get_teams(contract, hands)

    assert player_team == [1]
    assert sorted(opponent_team) == [0, 2, 3]


# ---------------------------------------------------------------------------
# Ramsch — must raise
# ---------------------------------------------------------------------------


def test_ramsch_raises_not_implemented():
    contract = GameModeFactory.create_ramsch()
    hands = [empty_hand()] * 4

    with pytest.raises(NotImplementedError):
        get_teams(contract, hands)


# ---------------------------------------------------------------------------
# Team sizes are always correct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "contract_fn, expected_player_size",
    [
        (lambda: GameModeFactory.create_wenz(caller_id=0), 1),
        (lambda: GameModeFactory.create_geier(caller_id=0), 1),
        (lambda: GameModeFactory.create_solo(caller_id=0, suit=Suit.Herz), 1),
    ],
)
def test_team_sizes_sum_to_four(contract_fn, expected_player_size):
    contract = contract_fn()
    hands = [empty_hand()] * 4

    player_team, opponent_team = get_teams(contract, hands)

    assert len(player_team) == expected_player_size
    assert len(player_team) + len(opponent_team) == 4
    assert set(player_team) | set(opponent_team) == {0, 1, 2, 3}
    assert set(player_team) & set(opponent_team) == set()
