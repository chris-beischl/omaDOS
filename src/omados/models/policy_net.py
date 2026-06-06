import torch
from torch import nn


class PolicyNet(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        layers.append(nn.Linear(obs_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, 32))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # returns raw logits over 32 cards
        result: torch.Tensor = self.net(obs)
        return result
