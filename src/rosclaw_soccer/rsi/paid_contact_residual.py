"""Plastic full29-joint contact correction over an immutable learned motor.

Original actor input remains160. The new residual input is288: normalized160
measured observations plus128 causal features of that SAME frozen recurrent
actor. No case/source/goal-success/teacher-IK label is an inference input.
"""

from typing import Any

import torch
from torch import nn


class ContactResidual(nn.Module):
    def __init__(self, *, bound_radians: float = 0.2) -> None:
        super().__init__()
        if bound_radians != 0.2:
            raise ValueError("declared teacher-matched .2rad per-joint residual bound required")
        self.bound_radians = bound_radians
        self.net = nn.Sequential(
            nn.Linear(288, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 29)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if (
            features.ndim != 2
            or features.shape[1] != 288
            or features.dtype != torch.float32
            or not torch.isfinite(features).all()
        ):
            raise ValueError("finite explicit batch288 float32 residual features required")
        return self.bound_radians * torch.tanh(self.net(features))


class FrozenContactResidualActor(nn.Module):
    def __init__(self, base: Any, residual: ContactResidual, scale: torch.Tensor) -> None:
        super().__init__()
        if (
            scale.shape != (29,)
            or scale.dtype != torch.float32
            or not torch.isfinite(scale).all()
            or not (scale > 0).all()
        ):
            raise ValueError("exact original positive policy-order29 action scales required")
        self.base, self.residual = base, residual
        self.register_buffer("action_scale", scale.detach().clone())
        self.base.requires_grad_(False)
        self.base.eval()

    def forward(self, obs, state):
        nominal, next_state, hidden = self.base.forward_with_features(obs, state)
        feature = torch.cat([(obs - self.base.mean) / self.base.denominator, hidden], dim=-1)
        delta = self.residual(feature.reshape(-1, 288)).reshape(*nominal.shape)
        return nominal + delta / self.action_scale, next_state
