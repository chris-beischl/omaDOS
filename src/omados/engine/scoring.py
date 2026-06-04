from dataclasses import dataclass

import torch

from omados.engine.tricks import Trick


@dataclass
class GameOutcome:
    player_team_won: bool
    schneider: bool
    schwarz: bool
    caller_team_points: int
    # TODO: laufende, tout, etc.


# TODO: add Tout/Stoß support once bidding phase is complete
def determine_outcome(
    scores: torch.Tensor,  # shape (4,) — points per player
    player_team: list[int],
    tricks_per_player: dict[int, list[Trick]],  # you need trick counts, not just points
) -> GameOutcome:
    player_team_score = scores[player_team].sum()
    opponent_team_score = 120 - player_team_score

    player_team_won = player_team_score > 60
    schneider = (
        (opponent_team_score <= 29) if player_team_won else (player_team_score <= 30)
    )

    player_team_tricks = []
    for pid in player_team:
        player_team_tricks.extend(tricks_per_player[pid])

    schwarz = (
        len(player_team_tricks) == 8
        if player_team_won
        else len(player_team_tricks) == 0
    )

    return GameOutcome(
        player_team_won=bool(player_team_won),
        schneider=bool(schneider),
        schwarz=schwarz,
        caller_team_points=int(player_team_score.item()),
    )
