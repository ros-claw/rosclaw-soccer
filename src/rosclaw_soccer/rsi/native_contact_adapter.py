"""SIM hypothesis: frozen foundation plus bounded near-ball neural plasticity.

The adapter starts with exactly zero influence. When measured pelvis-to-ball
planar distance >=0.7m, or ball perception is absent, output stays the immutable
foundation even after adapter learning. No source/case/outcome lookup is used.
This is architecture, not trained skill evidence or hardware authorization.
"""

from typing import Any, cast

import torch
from torch import nn


class NativeResidualContactActor(nn.Module):
    reference_features = 79
    near_radius_m = 0.35
    far_radius_m = 0.7
    maximum_normalized_residual = 0.2

    def __init__(self, base: Any):
        super().__init__()
        if base.reference_features != 79:
            raise ValueError("exact measured177 foundation required")
        self.base = base.eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        head = nn.Linear(64, 29)
        self.adapter = nn.Sequential(
            nn.Linear(177, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), head
        )
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)

    @property
    def prior_mean(self) -> torch.Tensor:
        return cast(torch.Tensor, self.base.prior_mean)

    @property
    def prior_denominator(self) -> torch.Tensor:
        return cast(torch.Tensor, self.base.prior_denominator)

    def measured_gate(self, observation: torch.Tensor) -> torch.Tensor:
        if (
            observation.ndim != 2
            or observation.shape[1] != 177
            or observation.dtype != torch.float32
            or not torch.isfinite(observation).all()
        ):
            raise ValueError("finite measured float32 native177 inputs required")
        distance = observation[:, 168:170].norm(dim=1)
        fraction = (
            (self.far_radius_m - distance) / (self.far_radius_m - self.near_radius_m)
        ).clamp(0, 1)
        present = observation[:, 168:177].abs().sum(dim=1) > 0
        return cast(torch.Tensor, fraction.square() * (3 - 2 * fraction) * present)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        gate = self.measured_gate(observation)
        foundation = self.base(observation)
        normalized = torch.cat(
            ((observation[:, :98] - self.prior_mean) / self.prior_denominator, observation[:, 98:]),
            dim=1,
        )
        residual = self.maximum_normalized_residual * self.adapter(normalized).tanh()
        output = foundation + gate[:, None] * residual
        if not torch.isfinite(output).all():
            raise ValueError("nonfinite contact adapter output")
        return cast(torch.Tensor, output)
