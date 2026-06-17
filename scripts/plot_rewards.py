"""
Plot a reward curve saved from a training run.

Usage:
    # After training, pipe rewards to a file:
    uv run python scripts/train_reinforce.py --out policy_net.pt 2>&1 | tee rewards.log

    # Or save rewards explicitly then plot:
    uv run python scripts/plot_rewards.py rewards.npy

    # Plot with a custom rolling window:
    uv run python scripts/plot_rewards.py rewards.npy --window 200
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(
    rewards: list[float],
    window: int = 100,
    title: str = "REINFORCE Training",
    save_path: str | None = None,
) -> None:
    rewards_np = np.array(rewards)
    episodes = np.arange(len(rewards_np))

    # Rolling mean
    kernel = np.ones(window) / window
    rolling_mean = np.convolve(rewards_np, kernel, mode="valid")
    rolling_episodes = episodes[window - 1 :]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, rewards_np, alpha=0.2, color="steelblue", label="per-episode")
    ax.plot(
        rolling_episodes,
        rolling_mean,
        color="steelblue",
        label=f"rolling mean ({window})",
    )
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rewards_file", help=".npy file containing rewards array")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument(
        "--out", type=str, default=None, help="Save PNG instead of showing"
    )
    args = parser.parse_args()

    rewards = np.load(args.rewards_file).tolist()
    plot_rewards(rewards, window=args.window, save_path=args.out)
