
"""Configurable feed-forward Q-network for DQN."""

import torch
from torch import nn
from torch.nn import functional as F


class DQN(nn.Module):
    """Maps a CartPole observation to one Q-value per discrete action."""

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        hidden_units: int = 128,
    ):
        super().__init__()

        self.hidden_units = hidden_units

        self.layer1 = nn.Linear(n_observations, hidden_units)
        self.layer2 = nn.Linear(hidden_units, hidden_units)
        self.layer3 = nn.Linear(hidden_units, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
