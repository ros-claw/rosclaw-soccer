"""Research-only masked neural refinement; no native checkpoint admission yet."""

import copy
from typing import Any

import torch
from torch import nn

from rosclaw_soccer.rsi.feature_support_gate import FeatureSupportGate
from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter


class SupportedRefinementMotor(nn.Module):
    """Reuse a learned neural head only inside independently qualified support.

    Outside support, return the entire acquired parent's actual output exactly.
    Inside, keep the ORIGINAL total +/-0.2rad constraint, never extra authority.
    No optimizer is supplied: reusing trained weights is not another update.
    """

    def __init__(self, refinement: RefinementMotorAdapter, support: FeatureSupportGate):
        super().__init__()
        if (
            type(refinement) is not RefinementMotorAdapter
            or type(support) is not FeatureSupportGate
        ):
            raise ValueError(
                "explicit learned refinement and qualified data-support types required"
            )
        self.refinement = copy.deepcopy(refinement).requires_grad_(False).eval()
        self.support = copy.deepcopy(support).requires_grad_(False).eval()
        self.eval()

    def train(self, mode: bool = True) -> "SupportedRefinementMotor":
        if mode:
            raise ValueError("read-only support-masked research candidate; no fitting interface")
        super().train(False)
        return self

    def initial_state(self, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        return self.refinement.initial_state(batch=batch)

    def adapt_features(
        self, features: torch.Tensor, parent_delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mix, _ = self.refinement.adapt_features(features, parent_delta)
        supported_mix = mix * self.support(features)
        total = torch.clamp(
            parent_delta.detach() * (1 - supported_mix.abs()) + 0.2 * supported_mix, -0.2, 0.2
        )
        return supported_mix, total

    def forward_with_evidence(
        self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
        parent_mean, after, e = self.refinement.own_features(obs, state)
        activation = self.support(e["critic_features"])
        requested_mix, _ = self.refinement.adapt_features(
            e["critic_features"], e["final_mean_delta_radians"]
        )
        mix = requested_mix * activation
        total = torch.clamp(e["final_mean_delta_radians"] * (1 - mix.abs()) + 0.2 * mix, -0.2, 0.2)
        scale = self.refinement.parent.memory.parent.get_buffer("action_scale")
        mean = torch.where(activation == 0, parent_mean, e["reference_nominal"] + total / scale)
        return (
            mean,
            after,
            dict(
                **{
                    k: v
                    for k, v in e.items()
                    if k not in ("final_mean_delta_radians", "physical_retention_proven")
                },
                refinement_parent_mean=parent_mean,
                support_activation=activation,
                supported_refinement_mix=mix,
                final_mean_delta_radians=total,
                physical_retention_proven=False,
            ),
        )

    def forward(
        self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        mean, after, _ = self.forward_with_evidence(obs, state)
        return mean, after
