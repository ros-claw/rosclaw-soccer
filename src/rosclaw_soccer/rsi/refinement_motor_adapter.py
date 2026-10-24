"""SIM research: freeze the WHOLE acquired motor and learn within its old envelope.

The new head has access to the entire original +/-0.2 total-delta interval,
not an extra +/-0.2 around the parent. Zero output reproduces the full parent,
including its learned head, but says nothing about retention AFTER training.
No checkpoint admission, physical executor, or deployment authority is provided.
"""

import copy
from typing import Any

import torch
from torch import nn

from rosclaw_soccer.rsi.consolidated_motor_adapter import ConsolidatedMotorAdapter


class RefinementMotorAdapter(nn.Module):
    hidden_width = 256
    simulation_only = True
    policy_domain = "research.consolidated_motor_refinement.v1"

    def __init__(self, parent: ConsolidatedMotorAdapter) -> None:
        super().__init__()
        if type(parent) is not ConsolidatedMotorAdapter:
            raise ValueError("exact whole acquired motor required")
        self.parent = copy.deepcopy(parent).requires_grad_(False).eval()
        output = nn.Linear(128, 29)
        self.head = nn.Sequential(nn.Linear(288, 128), nn.Tanh(), output)
        parameter = next(parent.parameters())
        self.head.to(device=parameter.device, dtype=parameter.dtype)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.train(False)

    def train(self, mode: bool = True) -> "RefinementMotorAdapter":
        super().train(mode)
        self.parent.eval()
        return self

    def initial_state(self, *, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        return self.parent.initial_state(batch=batch)

    def own_features(
        self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
        with torch.no_grad():
            return self.parent.forward_with_evidence(obs, state)

    def adapt_features(
        self, features: torch.Tensor, parent_delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            features.ndim != 3
            or features.shape[-1] != 288
            or not 1 <= features.shape[0] <= 512
            or not 1 <= features.shape[1] <= 64
            or features.dtype != torch.float32
            or parent_delta.shape != (*features.shape[:2], 29)
            or parent_delta.dtype != features.dtype
            or parent_delta.device != features.device
            or not torch.isfinite(features).all()
            or not torch.isfinite(parent_delta).all()
            or (parent_delta.abs() > 0.2).any()
        ):
            raise ValueError("finite own288 features and bounded original parent delta required")
        logits = self.head(features.detach())
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite refinement head rejected")
        mix = torch.tanh(logits)
        # Convex interpolation from the parent to either OLD boundary. The
        # abs gradient is zero at mix=0: both directions retain .2 slope even
        # if the parent is saturated, unlike a one-sided remaining-budget gate.
        total = torch.clamp(parent_delta.detach() * (1 - mix.abs()) + 0.2 * mix, -0.2, 0.2)
        return mix, total

    def forward_with_evidence(
        self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
        parent_mean, after, evidence = self.own_features(obs, state)
        mix, total = self.adapt_features(
            evidence["critic_features"], evidence["final_mean_delta_radians"]
        )
        scale = self.parent.memory.parent.get_buffer("action_scale")
        mean = evidence["reference_nominal"] + total / scale
        return (
            mean,
            after,
            dict(
                **{
                    k: v
                    for k, v in evidence.items()
                    if k not in ("final_mean_delta_radians", "physical_retention_proven")
                },
                refinement_parent_mean=parent_mean,
                refinement_mix=mix,
                refinement_delta_radians=total - evidence["final_mean_delta_radians"],
                final_mean_delta_radians=total,
                physical_retention_proven=False,
            ),
        )

    def forward(
        self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        mean, after, _ = self.forward_with_evidence(obs, state)
        return mean, after

    def plastic_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(p for p in self.head.parameters() if p.requires_grad)
