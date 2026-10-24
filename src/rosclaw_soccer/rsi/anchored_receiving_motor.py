"""Trainable WHOLE velocity MLP with an exact frozen deployment anchor.

Prototype only: no collector, optimizer, checkpoint admission, world or device
transport. The caller must recompute the ONNX reference on the SAME actual98
observation. All eight plastic MLP tensors can learn, not just an output head.
This remains bounded joint POSITION control, not direct joint torque learning.
"""

import copy
from typing import Any

import torch
from torch import nn

from rosclaw_soccer.rsi.bounded_motor_exploration import motor_box
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor


class AnchoredReceivingMotor(nn.Module):
    action_scale: torch.Tensor
    cue_scale: torch.Tensor
    simulation_only = True
    policy_domain = "research.receiver_whole_velocity_actor.v1"
    observation_features = 112
    cue_features = 14

    def __init__(self, initial: VelocityPriorActor, action_scale: torch.Tensor) -> None:
        super().__init__()
        if (
            type(initial) is not VelocityPriorActor
            or initial.reference_features != 14
            or action_scale.shape != (29,)
            or action_scale.dtype != torch.float32
            or not torch.isfinite(action_scale).all()
            or not (action_scale > 0).all()
            or any(
                p.dtype != torch.float32
                or p.device != action_scale.device
                or not torch.isfinite(p).all()
                for p in initial.parameters()
            )
        ):
            raise ValueError(
                "exact velocity98+causal14 MLP and unchanged positive29 scales required"
            )
        self.plastic = copy.deepcopy(initial).requires_grad_(True)
        self.baseline = copy.deepcopy(initial).requires_grad_(False).eval()
        self.register_buffer("action_scale", action_scale.detach().clone())
        self.register_buffer(
            "cue_scale",
            torch.tensor(
                [4.0, 4.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0],
                dtype=torch.float32,
                device=action_scale.device,
            ),
        )
        self.train(False)

    def train(self, mode: bool = True) -> "AnchoredReceivingMotor":
        super().train(mode)
        self.baseline.eval()
        return self

    def forward_with_evidence(
        self, observations: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if (
            observations.ndim != 2
            or observations.shape[1] != 112
            or not 1 <= len(observations) <= 8192
            or observations.dtype != torch.float32
            or reference.shape != (len(observations), 29)
            or reference.dtype != torch.float32
            or reference.device != observations.device
            or reference.requires_grad
            or self.action_scale.device != observations.device
            or not torch.isfinite(observations).all()
            or not torch.isfinite(reference).all()
            or self.baseline.training
            or any(p.requires_grad for p in self.baseline.parameters())
        ):
            raise ValueError(
                "finite actual own112 inputs and same-observation frozen ONNX reference required"
            )
        inputs = torch.cat((observations[:, :98], observations[:, 98:] / self.cue_scale), dim=-1)
        proposed = self.plastic(inputs)
        with torch.no_grad():
            original = self.baseline(inputs)
        if not torch.isfinite(proposed).all() or not torch.isfinite(original).all():
            raise ValueError("nonfinite whole plastic motor output")
        requested = (proposed - original) * self.action_scale
        bounded = 0.2 * torch.tanh(requested / 0.2)
        raw = reference + bounded / self.action_scale
        lower, upper = motor_box(reference, self.action_scale)
        mean = torch.maximum(torch.minimum(raw, upper.float()), lower.float())
        return mean, dict(
            requested_delta_radians=requested,
            realized_delta_radians=(mean.double() - reference.double())
            * self.action_scale.double(),
            critic_features=observations.detach(),
            physical_retention_proven=False,
            neural_learning_proven=False,
            optimizer_steps_executed=0,
        )

    def forward(self, observations: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return self.forward_with_evidence(observations, reference)[0]

    def plastic_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.plastic.parameters())
