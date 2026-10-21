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
        self.gate = nn.Sequential(nn.Linear(288, 64), nn.ELU(), nn.Linear(64, 1))
        output, gate_output = self.net[-1], self.gate[-1]
        if not isinstance(output, nn.Linear) or not isinstance(gate_output, nn.Linear):
            raise ValueError("exact separate linear residual and gate heads required")
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        nn.init.zeros_(gate_output.weight)
        nn.init.constant_(gate_output.bias, -4.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        delta, logits = self.forward_with_gate(features)
        # A learned reject option: unsupported contexts retain the base motor
        # EXACTLY, not merely a small unrequested perturbation. No case labels.
        return delta if self.training else torch.where(logits >= 0, delta, torch.zeros_like(delta))

    def forward_with_gate(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            features.ndim != 2
            or features.shape[1] != 288
            or features.dtype != torch.float32
            or not torch.isfinite(features).all()
        ):
            raise ValueError("finite explicit batch288 float32 residual features required")
        logits = self.gate(features)
        return self.bound_radians * torch.tanh(self.net(features)) * torch.sigmoid(logits), logits


class DecoupledContactResidual(ContactResidual):
    """Same .2rad authority, binary rejection without probability attenuation."""

    def forward_with_gate(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            features.ndim != 2
            or features.shape[1] != 288
            or features.dtype != torch.float32
            or not torch.isfinite(features).all()
        ):
            raise ValueError("finite explicit batch288 float32 residual features required")
        return self.bound_radians * torch.tanh(self.net(features)), self.gate(features)


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

    def forward(
        self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        nominal, next_state, hidden = self.base.forward_with_features(obs, state)
        feature = torch.cat([(obs - self.base.mean) / self.base.denominator, hidden], dim=-1)
        delta = self.residual(feature.reshape(-1, 288)).reshape(*nominal.shape)
        return nominal + delta / self.action_scale, next_state


def residual_config(variant: int, *, projected: bool = False) -> dict[str, Any]:
    if type(variant) is not int or variant not in range(4):
        raise ValueError("one predeclared residual training seed or retention ablation required")
    config = dict(
        protocol="frozen-contact-residual-v1",
        variant=variant,
        seed=20261104 + (variant if variant < 3 else 0),
        epochs=256,
        lr=0.0003,
        retention_weight=5.0 if variant < 3 else 1.0,
        active_weight=5.0,
        gate_loss_weight=0.001,
        gate_positive_weight=5.0,
        hard_gate_at_inference=True,
        inference_gate_probability_threshold=0.5,
        max_grad_norm=1.0,
        bound_radians=0.2,
        residual_features=288,
        original_actor_features=160,
        memory_handoff="reset",
        frozen_base=True,
        unique_training_rows=3810,
        rescue_cases=[0, 1, 4, 8, 9, 15],
        old_good_zero_cases=[2, 3, 5, 6, 7, 11, 12, 13],
        failed_body_only_zero_cases=[10, 14],
        rescue_rows=1474,
        zero_retention_rows=2336,
        minibatches_per_epoch=4,
        rescue_batch_sizes=[2, 2, 1, 1],
        zero_batch_sizes=[3, 3, 2, 2],
        DEV_episodes_used=0,
        runtime_case_lookup=False,
        loss_units="joint_residual_radians_squared_plus_gate_BCE",
    )
    if projected:
        config.update(
            protocol="projected-contact-residual-v2",
            seed=20261107 + (variant if variant < 3 else 0),
            retention_weight=5.0,
            gate_projection=variant < 3,
            gate_retention_margin=0.1,
            max_projection_halvings=8,
            hard_negative_weight=0.005,
            hard_negative_margin=0.5,
            grown_positive_cases=[1, 8, 9, 15],
            unfinished_teacher_cases=[0, 4],
            grown_replay_rows=925,
            unfinished_teacher_rows=549,
            warmstart="qualified_previous_joint_delta_head_fresh_rejecting_gate",
            negative_support="all_zero_target_frames_in_all16_episodes",
        )
    return config
