"""Experimental target-conditioned joint/reference actor, not torque control.

Keep the qualified 133-feature motor path intact at initialization. Three
extra inputs are target-ray XY in that same frame and the previous reference
heading offset. Output is 29 motor residual logits plus one reference-heading
logit. The caller still owns contact gates, frame identity, motor envelopes,
physics stepping and independent validation. This does not activate a policy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic


def build_goal_reference_actor_critic(seed_state: Mapping[str, Any]) -> Any:
    """Initialize zero context/reference heads around an exact numeric motor seed.

    A separate contiguous motor input preserves its matrix shapes and arithmetic
    instead of silently padding its first weight matrix. A physical parity test
    is still required. Torch stays an optional, lazy training dependency.
    """
    import torch

    if (
        not isinstance(seed_state, Mapping)
        or not seed_state
        or any(
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or value.layout != torch.strided
            or not bool(torch.isfinite(value).all())
            for value in seed_state.values()
        )
    ):
        raise ValueError("finite CPU float32 numeric motor state required")
    motor = build_ball_residual_actor_critic()
    motor.load_state_dict(seed_state, strict=True)

    class GoalReferenceActorCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.actor, self.critic = motor.actor, motor.critic

            def head(inputs: int, outputs: int) -> Any:
                output = torch.nn.Linear(32, outputs)
                result = torch.nn.Sequential(torch.nn.Linear(inputs, 32), torch.nn.Tanh(), output)
                torch.nn.init.zeros_(output.weight)
                torch.nn.init.zeros_(output.bias)
                return result

            self.goal_actor = head(3, 29)
            self.goal_critic = head(3, 1)
            self.reference_actor = head(136, 1)
            self.logstd = torch.nn.Parameter(
                torch.cat((motor.logstd.detach().clone(), torch.tensor([-1.5])))
            )

        def forward(self, observation: Any) -> tuple[Any, Any]:
            if (
                not isinstance(observation, torch.Tensor)
                or observation.ndim != 2
                or observation.shape[1] != 136
                or not 1 <= observation.shape[0] <= 65536
                or not torch.is_floating_point(observation)
                or observation.layout != torch.strided
                or observation.dtype != self.logstd.dtype
                or observation.device != self.logstd.device
                or not bool(torch.isfinite(observation).all())
                or bool((observation.abs() > 10).any())
            ):
                raise ValueError("finite bounded (N, 136) goal-reference observations required")
            body = observation[:, :133].contiguous()
            context = observation[:, 133:].contiguous()
            joints = self.actor(body) + self.goal_actor(context)
            reference = self.reference_actor(observation)
            mean = torch.cat((joints, reference), dim=1)
            value = (self.critic(body) + self.goal_critic(context)).squeeze(-1)
            if not bool(torch.isfinite(mean).all() and torch.isfinite(value).all()):
                raise FloatingPointError("nonfinite goal-reference actor or critic output")
            return mean, value

    return GoalReferenceActorCritic()


@dataclass(frozen=True)
class ReferenceHeadingEnvelope:
    center_rad: float = -0.4
    maximum_offset_rad: float = 0.2
    maximum_step_rad: float = 0.02
    smoothing: float = 0.2
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        values = (self.center_rad, self.maximum_offset_rad, self.maximum_step_rad, self.smoothing)
        if (
            any(type(v) not in (float, int) or not math.isfinite(v) for v in values)
            or not 0 < self.maximum_offset_rad <= 0.6
            or abs(self.center_rad) + self.maximum_offset_rad > math.pi
            or not 0 < self.maximum_step_rad <= 0.03
            or not 0 < self.smoothing <= 1
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("bounded simulation-only reference-heading envelope required")


def advance_reference_heading(
    raw_action: Any, previous: Any, envelope: ReferenceHeadingEnvelope | None = None
) -> Any:
    """Bound reference input changes; never modify physical body orientation."""
    import torch

    if envelope is not None and not isinstance(envelope, ReferenceHeadingEnvelope):
        raise ValueError("declared reference-heading envelope required")
    active = envelope or ReferenceHeadingEnvelope()
    if (
        not isinstance(raw_action, torch.Tensor)
        or not isinstance(previous, torch.Tensor)
        or raw_action.ndim != 1
        or raw_action.shape != previous.shape
        or not 1 <= raw_action.shape[0] <= 4096
        or raw_action.device != previous.device
        or raw_action.dtype != previous.dtype
        or raw_action.dtype not in (torch.float32, torch.float64)
        or not bool(torch.isfinite(raw_action).all() and torch.isfinite(previous).all())
        or bool((previous < active.center_rad - active.maximum_offset_rad).any())
        or bool((previous > active.center_rad + active.maximum_offset_rad).any())
    ):
        raise ValueError("finite bounded matching reference-heading states required")
    desired = active.center_rad + active.maximum_offset_rad * torch.tanh(raw_action)
    return previous + torch.clamp(
        active.smoothing * (desired - previous),
        -active.maximum_step_rad,
        active.maximum_step_rad,
    )
