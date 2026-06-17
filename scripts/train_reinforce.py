"""
REINFORCE training script.

Usage:
    python scripts/train_reinforce.py
    python scripts/train_reinforce.py --episodes 5000 --batch-size 64
"""

import argparse
import logging

import numpy as np
import torch

from omados.agents.random import RandomAgent
from omados.agents.rl import OBS_SIZE, RLAgent
from omados.env.game import SchafkopfEnv
from omados.models.policy_net import PolicyNet
from omados.training.reinforce import train

logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Schafkopf RL agent.")
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=str, default="policy_net.pt")
    parser.add_argument(
        "--save-rewards", type=str, default=None, help="Save rewards to .npy file"
    )
    return parser.parse_args()


args = parse_args()

# --- Setup ---
model = PolicyNet(obs_size=OBS_SIZE, hidden_size=args.hidden_size, depth=args.depth)
rl_agent = RLAgent(player_id=0, model=model)
agents = [
    rl_agent,
    RandomAgent(player_id=1),
    RandomAgent(player_id=2),
    RandomAgent(player_id=3),
]
env = SchafkopfEnv(agents)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# --- Train ---
rewards = train(
    agent=rl_agent,
    env=env,
    n_episodes=args.episodes,
    optimizer=optimizer,
    gamma=args.gamma,
    baseline=not args.no_baseline,
    batch_size=args.batch_size,
)

# --- Save ---
torch.save(model.state_dict(), args.out)
print(
    f"\nTraining complete. Mean reward (last 500): "
    f"{sum(rewards[-500:]) / min(len(rewards), 500):.4f}"
)
print(f"Model saved to {args.out}")
if args.save_rewards:
    np.save(args.save_rewards, np.array(rewards))
    print(f"Rewards saved to {args.save_rewards}")
