"""评分网络：输入状态 + 动作特征，输出标量分数。"""
from __future__ import annotations

import torch
import torch.nn as nn


class ScoreNet(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 128),
            nn.ReLU(),
        )
        self.act_mlp = nn.Sequential(
            nn.Linear(act_dim, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, act: torch.Tensor):
        s = self.state_mlp(state)
        a = self.act_mlp(act)
        return self.head(torch.cat([s, a], dim=1)).squeeze(-1)
