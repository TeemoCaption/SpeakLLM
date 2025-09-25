"""Duplex controller predicting continue/hold/barge-in."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class DuplexControllerConfig:
    input_dim: int = 768
    hidden_dim: int = 512
    dropout: float = 0.1
    num_actions: int = 3


class DuplexController(nn.Module):
    def __init__(self, config: DuplexControllerConfig | None = None) -> None:
        super().__init__()
        self.config = config or DuplexControllerConfig()
        self.net = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim // 2, self.config.num_actions),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def sample_action(self, features: torch.Tensor, temperature: float = 1.0) -> Tuple[int, torch.Tensor]:
        logits = self.forward(features)
        probs = torch.softmax(logits / temperature, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return int(action.item()), probs
