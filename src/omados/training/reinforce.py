import copy
import logging

import torch
import torch.optim as optim

from omados.agents.rl import RLAgent
from omados.env.game import SchafkopfEnv

logger = logging.getLogger(__name__)


def train(
    agent: RLAgent,
    env: SchafkopfEnv,
    n_episodes: int,
    optimizer: optim.Optimizer,
    gamma: float = 1.0,
    baseline: bool = True,
) -> list[float]:
    """
    REINFORCE training loop.

    Runs n_episodes games, collects one trajectory per game, and updates
    the agent's policy after each episode.

    Args:
        agent:       The RLAgent being trained. All other seats are fixed opponents.
        env:         The SchafkopfEnv. The agent must be in env.agents.
        n_episodes:  Number of games to play.
        optimizer:   PyTorch optimizer for agent.model parameters.
        gamma:       Discount factor. 1.0 = no discounting (reward only at end).
        baseline:    If True, subtract a running mean reward baseline to reduce variance

    Returns:
        List of per-episode rewards for plotting.
    """
    reward_history: list[float] = []
    running_mean: float = 0.0

    agent.model.train()

    for episode in range(n_episodes):
        # Update frozen opponents every 100 episodes with a snapshot of current policy
        if episode % 100 == 0 and episode > 0:
            frozen_model = copy.deepcopy(agent.model)
            frozen_model.eval()
            for i in range(1, 4):
                env.agents[i] = RLAgent(player_id=i, model=frozen_model)
            logger.debug(f"Episode {episode}: Updated frozen opponents.")

        # Reset all agents' state for the new episode
        agent.trajectory = []
        for a in env.agents:
            a.memory = []

        result = env.play_game()

        if result is None:
            # Ramsch — no contract, skip episode
            logger.debug(f"Episode {episode}: Ramsch, skipping.")
            continue

        _, player_reward, opponent_reward, _, player_team, _ = result

        # Assign reward based on which team the agent was on
        reward = float(
            player_reward if agent.player_id in player_team else opponent_reward
        )
        reward_history.append(reward)

        # Skip update if agent never played (e.g. wasn't the RL agent in this game)
        if not agent.trajectory:
            continue

        # Baseline: subtract running mean to reduce variance
        if baseline:
            running_mean += (reward - running_mean) / (episode + 1)
            adjusted_reward = reward - running_mean
        else:
            adjusted_reward = reward

        # Compute discounted returns for each step.
        # Since reward is only given at end of game, G_t = gamma^(T-t) * R_T.
        T = len(agent.trajectory)
        returns = torch.tensor(
            [adjusted_reward * (gamma ** (T - 1 - t)) for t in range(T)],
            dtype=torch.float32,
        )

        # Stack log probs from trajectory: shape (T,)
        log_probs = torch.stack([step["log_prob"] for step in agent.trajectory])

        # REINFORCE loss: -E[log_prob * G]
        loss = -(log_probs * returns).sum()

        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        if episode % 100 == 0:
            mean_reward = sum(reward_history[-100:]) / max(
                len(reward_history[-100:]), 1
            )
            logger.info(
                f"Episode {episode:>6} | reward: {reward:+.2f} | "
                f"mean(last 100): {mean_reward:+.4f}"
            )

    return reward_history
