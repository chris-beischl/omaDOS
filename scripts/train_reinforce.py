"""
REINFORCE training script.

Usage:
    python scripts/train_reinforce.py
"""

import logging

import torch

from omados.agents.random import RandomAgent
from omados.agents.rl import OBS_SIZE, RLAgent
from omados.env.game import SchafkopfEnv
from omados.models.policy_net import PolicyNet
from omados.training.reinforce import train

logging.basicConfig(level=logging.INFO)

# --- Hyperparameters ---
N_EPISODES = 10_000
HIDDEN_SIZE = 512
DEPTH = 3
LR = 1e-3
GAMMA = 1.0
BASELINE = True

# --- Setup ---
model = PolicyNet(obs_size=OBS_SIZE, hidden_size=HIDDEN_SIZE, depth=DEPTH)
rl_agent = RLAgent(player_id=0, model=model)
agents = [
    rl_agent,
    RandomAgent(player_id=1),
    RandomAgent(player_id=2),
    RandomAgent(player_id=3),
]
env = SchafkopfEnv(agents)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Train ---
rewards = train(
    agent=rl_agent,
    env=env,
    n_episodes=N_EPISODES,
    optimizer=optimizer,
    gamma=GAMMA,
    baseline=BASELINE,
)

# --- Save ---
torch.save(model.state_dict(), "policy_net.pt")
print(
    f"\nTraining complete. Mean reward (last 500): "
    f"{sum(rewards[-500:]) / min(len(rewards), 500):.4f}"
)
print("Model saved to policy_net.pt")
