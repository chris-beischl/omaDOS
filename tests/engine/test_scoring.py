"""
Tests for determine_outcome and get_rewards in scoring.py.
Covers: win/loss, schneider, schwarz, reward sign and magnitude.
"""

import pytest
import torch

from omados.engine.scoring import GameOutcome, determine_outcome, get_rewards
from omados.engine.tricks import Trick

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_scores(*player_points: int) -> torch.Tensor:
    """Build a 4-element scores tensor from per-player points."""
    assert len(player_points) == 4
    return torch.tensor(player_points, dtype=torch.float)


def make_tricks_per_player(
    trick_counts: dict[int, int],
) -> dict[int, list[Trick]]:
    """Build a tricks_per_player dict using dummy Trick objects."""
    result: dict[int, list[Trick]] = {pid: [] for pid in range(4)}
    for pid, count in trick_counts.items():
        for _ in range(count):
            t = Trick(lead_player_id=pid)
            t.add_card("G7")  # minimal valid trick
            result[pid].append(t)
    return result


# ---------------------------------------------------------------------------
# determine_outcome — win / loss
# ---------------------------------------------------------------------------


def test_player_team_wins_exactly_61():
    """61 points is the minimum to win."""
    player_team = [0, 1]
    scores = make_scores(40, 21, 30, 29)  # team [0,1] = 61
    tricks = make_tricks_per_player({0: 3, 1: 2, 2: 2, 3: 1})

    outcome = determine_outcome(scores, player_team, tricks)

    assert outcome.player_team_won is True
    assert outcome.caller_team_points == 61


def test_player_team_loses_exactly_60():
    """60 points is not enough — opponent wins."""
    player_team = [0, 1]
    scores = make_scores(35, 25, 40, 20)  # team [0,1] = 60
    tricks = make_tricks_per_player({0: 2, 1: 2, 2: 3, 3: 1})

    outcome = determine_outcome(scores, player_team, tricks)

    assert outcome.player_team_won is False
    assert outcome.caller_team_points == 60


def test_solo_caller_wins():
    """Solo: caller is alone in player_team."""
    player_team = [2]
    scores = make_scores(20, 25, 75, 0)  # caller (P2) has 75
    tricks = make_tricks_per_player({2: 5, 0: 2, 1: 1})

    outcome = determine_outcome(scores, player_team, tricks)

    assert outcome.player_team_won is True
    assert outcome.caller_team_points == 75


def test_solo_caller_loses():
    player_team = [2]
    scores = make_scores(40, 30, 50, 0)  # caller (P2) has 50 — loses
    tricks = make_tricks_per_player({2: 3, 0: 3, 1: 2})

    outcome = determine_outcome(scores, player_team, tricks)

    assert outcome.player_team_won is False
    assert outcome.caller_team_points == 50


# ---------------------------------------------------------------------------
# determine_outcome — schneider
# ---------------------------------------------------------------------------


def test_no_schneider_normal_win():
    """Player team wins with 70 points — not schneider territory."""
    player_team = [0, 1]
    scores = make_scores(40, 30, 30, 20)  # team = 70, opponents = 50
    tricks = make_tricks_per_player({0: 3, 1: 3, 2: 1, 3: 1})

    outcome = determine_outcome(scores, player_team, tricks)

    assert outcome.player_team_won is True
    assert outcome.schneider is False


def test_schneider_winner_has_90():
    """Player team wins with 90 points → schneider (opponents have exactly 30 ≤ 30)."""
    player_team = [0, 1]
    scores = make_scores(50, 40, 20, 10)  # team = 90, opponents = 30
    tricks = make_tricks_per_player({0: 4, 1: 3, 2: 1})

    outcome = determine_outcome(scores, player_team, tricks)

    assert outcome.player_team_won is True
    assert outcome.schneider is True  # losers have 30 → schneider


def test_schneider_winner_has_91():
    """91 points also triggers schneider."""
    player_team = [0, 1]
    scores = make_scores(50, 41, 20, 9)  # team = 91, opponents = 29
    tricks = make_tricks_per_player({0: 4, 1: 3, 2: 1})

    outcome = determine_outcome(scores, player_team, tricks)

    assert outcome.schneider is True


def test_schneider_threshold_not_met_89():
    """89 points → no schneider (opponent has 31)."""
    player_team = [0, 1]
    scores = make_scores(50, 39, 20, 11)  # team = 89, opponents = 31
    tricks = make_tricks_per_player({0: 4, 1: 3, 2: 1})

    outcome = determine_outcome(scores, player_team, tricks)

    assert outcome.schneider is False


def test_schneider_loser_has_30_or_fewer():
    """Opponents win with 90+. Losing team has 30 → schneider for opponents."""
    player_team = [0, 1]
    scores = make_scores(15, 15, 50, 40)  # team = 30, opponents = 90
    tricks = make_tricks_per_player({2: 4, 3: 4})

    outcome = determine_outcome(scores, player_team, tricks)

    assert outcome.player_team_won is False
    assert outcome.schneider is True


def test_schneider_loser_has_31_no_schneider():
    player_team = [0, 1]
    scores = make_scores(20, 11, 50, 39)  # team = 31, opponents = 89
    tricks = make_tricks_per_player({2: 4, 3: 3, 0: 1})

    outcome = determine_outcome(scores, player_team, tricks)

    assert outcome.player_team_won is False
    assert outcome.schneider is False


# ---------------------------------------------------------------------------
# determine_outcome — schwarz
# ---------------------------------------------------------------------------


def test_schwarz_winner_takes_all_tricks():
    """Player team wins all 8 tricks → schwarz."""
    player_team = [0, 1]
    scores = make_scores(60, 60, 0, 0)  # team = 120
    tricks = make_tricks_per_player({0: 4, 1: 4})

    outcome = determine_outcome(scores, player_team, tricks)

    assert outcome.player_team_won is True
    assert outcome.schwarz is True
    assert outcome.schneider is True  # 120 points → always schneider too


def test_schwarz_loser_wins_no_tricks():
    """Player team wins 0 tricks → schwarz against them."""
    player_team = [0, 1]
    scores = make_scores(0, 0, 70, 50)
    tricks = make_tricks_per_player({2: 5, 3: 3})

    outcome = determine_outcome(scores, player_team, tricks)

    assert outcome.player_team_won is False
    assert outcome.schwarz is True


def test_no_schwarz_winner_misses_one_trick():
    """Player team wins 7 of 8 tricks — not schwarz."""
    player_team = [0, 1]
    scores = make_scores(50, 50, 15, 5)  # team = 100, but opponents took 1 trick
    tricks = make_tricks_per_player({0: 4, 1: 3, 2: 1})

    outcome = determine_outcome(scores, player_team, tricks)

    assert outcome.player_team_won is True
    assert outcome.schwarz is False


def test_no_schwarz_loser_has_one_trick():
    player_team = [0, 1]
    scores = make_scores(5, 0, 70, 45)  # team = 5, took 1 trick
    tricks = make_tricks_per_player({2: 4, 3: 3, 0: 1})

    outcome = determine_outcome(scores, player_team, tricks)

    assert outcome.player_team_won is False
    assert outcome.schwarz is False


# ---------------------------------------------------------------------------
# get_rewards
# ---------------------------------------------------------------------------


def test_rewards_normal_win():
    outcome = GameOutcome(
        player_team_won=True, schneider=False, schwarz=False, caller_team_points=70
    )
    player_r, opp_r = get_rewards(outcome)

    assert player_r == pytest.approx(1.0)
    assert opp_r == pytest.approx(-1.0)


def test_rewards_normal_loss():
    outcome = GameOutcome(
        player_team_won=False, schneider=False, schwarz=False, caller_team_points=50
    )
    player_r, opp_r = get_rewards(outcome)

    assert player_r == pytest.approx(-1.0)
    assert opp_r == pytest.approx(1.0)


def test_rewards_schneider_win():
    outcome = GameOutcome(
        player_team_won=True, schneider=True, schwarz=False, caller_team_points=90
    )
    player_r, opp_r = get_rewards(outcome)

    assert player_r == pytest.approx(1.5)
    assert opp_r == pytest.approx(-1.5)


def test_rewards_schneider_loss():
    outcome = GameOutcome(
        player_team_won=False, schneider=True, schwarz=False, caller_team_points=30
    )
    player_r, opp_r = get_rewards(outcome)

    assert player_r == pytest.approx(-1.5)
    assert opp_r == pytest.approx(1.5)


def test_rewards_schwarz_win():
    outcome = GameOutcome(
        player_team_won=True, schneider=True, schwarz=True, caller_team_points=120
    )
    player_r, opp_r = get_rewards(outcome)

    assert player_r == pytest.approx(2.0)
    assert opp_r == pytest.approx(-2.0)


def test_rewards_schwarz_loss():
    outcome = GameOutcome(
        player_team_won=False, schneider=True, schwarz=True, caller_team_points=0
    )
    player_r, opp_r = get_rewards(outcome)

    assert player_r == pytest.approx(-2.0)
    assert opp_r == pytest.approx(2.0)


def test_rewards_zero_sum():
    """Rewards must always sum to zero."""
    cases = [
        GameOutcome(True, False, False, 70),
        GameOutcome(False, False, False, 50),
        GameOutcome(True, True, False, 90),
        GameOutcome(False, True, True, 0),
    ]
    for outcome in cases:
        player_r, opp_r = get_rewards(outcome)
        assert player_r + opp_r == pytest.approx(0.0), (
            f"Rewards not zero-sum for {outcome}"
        )
