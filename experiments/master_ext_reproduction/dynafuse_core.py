"""Minimal neural components required by the final DynaFuse pipeline.

Extracted from local compatibility modules so the release package does not
depend on model-screening or discarded-candidate scripts.
"""
from __future__ import annotations

import torch
from torch import nn

from master import TAttention, TemporalAttention


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = pred.square().sum().sqrt() * target.square().sum().sqrt()
    return 1.0 - (pred * target).sum() / denom.clamp_min(1e-8)


class MasterTemporalResidual(nn.Module):
    def __init__(self, d_model: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(158, d_model)
        self.temporal = TAttention(d_model=d_model, nhead=2, dropout=dropout)
        self.pool = TemporalAttention(d_model=d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 32), nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, stock: torch.Tensor) -> torch.Tensor:
        z = self.temporal(self.input_proj(stock))
        return self.head(self.pool(z)).squeeze(-1)
