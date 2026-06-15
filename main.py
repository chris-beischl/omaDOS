"""Play a single demo game between four RandomAgents and print the outcome.

Run:
    uv run python main.py

To train the RL agent via REINFORCE self-play, see scripts/train_reinforce.py.
"""

import logging

from omados.agents.random import RandomAgent
from omados.env.game import SchafkopfEnv


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    agents = [RandomAgent(player_id=i) for i in range(4)]
    env = SchafkopfEnv(agents)

    result = env.play_game()
    if result is None:
        print("Everyone passed -> Ramsch (not yet implemented).")
        return

    outcome, player_reward, opponent_reward, contract, player_team, opponent_team = (
        result
    )
    print(f"Contract:      {contract.spielart.value}")
    print(f"Player team:   {player_team}  (reward {player_reward:+.1f})")
    print(f"Opponent team: {opponent_team}  (reward {opponent_reward:+.1f})")
    print(f"Outcome:       {outcome}")


if __name__ == "__main__":
    main()
