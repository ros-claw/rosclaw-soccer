"""Research prototype: consolidate the WHOLE learned motor, fit only a new head.

Zero-head identity is an algebraic/software property, NOT physical retention
after training. This module grants no checkpoint admission, promotion, or
hardware authority. Own recurrent features are computed from actual observation
history, never supplied by a teacher or indexed by a course/task identifier.
"""

import copy
import math
from typing import Any

import torch
from torch import nn

from rosclaw_soccer.rsi.dual_recurrent_motor import DualRecurrentMotor


class ConsolidatedMotorAdapter(nn.Module):
    hidden_width = 256
    simulation_only = True
    policy_domain = "research.consolidated_motor_adapter.v1"

    def __init__(self, memory: DualRecurrentMotor, *, head_bound_rad: float = 0.04) -> None:
        super().__init__()
        if type(memory) is not DualRecurrentMotor:
            raise ValueError("exact learned dual motor memory required")
        if (
            type(head_bound_rad) not in (float, int)
            or not math.isfinite(head_bound_rad)
            or not 0 < head_bound_rad <= 0.2
        ):
            raise ValueError("explicit finite new-head bound within original authority required")
        self.head_bound_rad = float(head_bound_rad)
        # Do not freeze/mutate the caller's model or alias any of its storage.
        self.memory = copy.deepcopy(memory).requires_grad_(False).eval()
        output = nn.Linear(128, 29)
        self.head = nn.Sequential(nn.Linear(288, 128), nn.Tanh(), output)
        reference = next(self.memory.parameters())
        self.head.to(device=reference.device, dtype=reference.dtype)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.train(False)

    def train(self, mode: bool = True) -> "ConsolidatedMotorAdapter":
        super().train(mode)
        self.memory.eval()
        return self

    def initial_state(self, *, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        return self.memory.initial_state(batch=batch)

    def own_features(
        self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
        """Advance the entire slow memory on the actual private input stream."""
        with torch.no_grad():
            mean, after, evidence = self.memory.forward_with_evidence(obs, state)
        return mean, after, evidence

    def adapt_features(
        self, features: torch.Tensor, memory_delta: torch.Tensor, scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Offline fit interface; callers must separately authenticate feature lineage.

        Returns a bounded added delta and total delta in radians. Frozen feature
        tensors cannot backpropagate into the consolidated memory. This method
        alone provides no evidence that the features came from a physical run.
        """
        if (
            features.ndim != 3
            or features.shape[-1] != 288
            or not 1 <= features.shape[0] <= 512
            or not 1 <= features.shape[1] <= 64
            or features.dtype != torch.float32
            or memory_delta.shape != (*features.shape[:2], 29)
            or memory_delta.dtype != features.dtype
            or memory_delta.device != features.device
            or scale.shape != (29,)
            or scale.dtype != features.dtype
            or scale.device != features.device
            or not torch.isfinite(features).all()
            or not torch.isfinite(memory_delta).all()
            or not torch.isfinite(scale).all()
            or not (scale > 0).all()
            or (memory_delta.abs() > 0.2).any()
        ):
            raise ValueError("finite own288 features and unchanged bounded memory delta required")
        logits = self.head(features.detach())
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite new motor head rejected")
        requested = self.head_bound_rad * torch.tanh(logits)
        old = memory_delta.detach()
        # Remaining authority relative to the ORIGINAL nominal motor, not an
        # extra +/-0.2 around the consolidated model. Zero is always admissible.
        added = torch.maximum(torch.minimum(requested, 0.2 - old), -0.2 - old)
        # Subtracting the remaining budget and adding it back is not closed
        # under float32: e.g. -.034487389 + -.165512621 is -.200000018.
        # Project the SUM too; do not loosen the original boundary by epsilon.
        total = torch.clamp(old + added, -0.2, 0.2)
        return added, total

    def forward_with_evidence(
        self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
        mean, after, evidence = self.own_features(obs, state)
        scale = self.memory.parent.get_buffer("action_scale")
        added, total = self.adapt_features(
            evidence["critic_features"],
            evidence["final_mean_delta_radians"],
            scale,
        )
        # Use the original nominal, not mean+added/scale, so the final sum
        # projection is actually executed. This is EXACTLY the slow motor's
        # original arithmetic at zero head: nominal + old_delta / scale.
        result = evidence["reference_nominal"] + total / scale
        return (
            result,
            after,
            dict(
                **{k: v for k, v in evidence.items() if k != "final_mean_delta_radians"},
                consolidated_mean=mean,
                adapter_delta_radians=total - evidence["final_mean_delta_radians"],
                adapter_delta_before_roundoff_projection=added,
                roundoff_projection_delta_radians=(
                    total - (evidence["final_mean_delta_radians"] + added)
                ),
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
