"""Ball-conditioned full-body residual learning contracts.

Soccer task code, not Core runtime authority. The actor adjusts a frozen
teacher's 29 joint targets; it is not an end-to-end torque controller. Training
rewards never constitute contact evidence or policy promotion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

BALL_RESIDUAL_OBSERVATION_SIZE = 133
BALL_RESIDUAL_ACTION_SIZE = 29


@dataclass(frozen=True)
class BallResidualEnvelope:
    maximum_offset_rad: float = 0.25
    maximum_step_rad: float = 0.025
    smoothing: float = 0.2
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        values = (self.maximum_offset_rad, self.maximum_step_rad, self.smoothing)
        if (
            not all(math.isfinite(v) for v in values)
            or not 0 < self.maximum_offset_rad <= 0.35
            or not 0 < self.maximum_step_rad <= 0.03
            or not 0 < self.smoothing <= 1
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("bounded simulation-only full-body residual envelope required")


def build_ball_residual_actor_critic() -> Any:
    """Lazy optional Torch backend; zero residual mean preserves the teacher.

    Observations: q-default(29), dq*0.1(29), gravity(3), root linear
    velocity(3), root angular velocity*0.2(3), ball-root translation(3),
    ball linear velocity(3), phase sin/cos(2), previous residual(29),
    teacher target-default(29). Current training uses a fixed field frame;
    this contract does not promise heading or body morphology invariance.
    """
    import torch

    class ActorCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

            def network(output: int) -> Any:
                return torch.nn.Sequential(
                    torch.nn.Linear(BALL_RESIDUAL_OBSERVATION_SIZE, 128),
                    torch.nn.Tanh(),
                    torch.nn.Linear(128, 128),
                    torch.nn.Tanh(),
                    torch.nn.Linear(128, output),
                )

            self.actor, self.critic = network(29), network(1)
            torch.nn.init.zeros_(self.actor[-1].weight)
            torch.nn.init.zeros_(self.actor[-1].bias)
            self.logstd = torch.nn.Parameter(torch.full((29,), -1.0))

        def forward(self, observation: Any) -> tuple[Any, Any]:
            if (
                observation.ndim != 2
                or observation.shape[1] != BALL_RESIDUAL_OBSERVATION_SIZE
                or not 1 <= observation.shape[0] <= 65536
                or not bool(torch.isfinite(observation).all())
                or bool((observation.abs() > 10).any())
            ):
                raise ValueError("finite bounded full-body actor observation required")
            mean, value = self.actor(observation), self.critic(observation).squeeze(-1)
            if not bool(torch.isfinite(mean).all() and torch.isfinite(value).all()):
                raise FloatingPointError("nonfinite full-body actor or critic output")
            return mean, value

    return ActorCritic()


def advance_ball_residual(
    raw_action: Any, previous: Any, envelope: BallResidualEnvelope | None = None
) -> Any:
    """Bound amplitude and per-control-tick change without touching world state."""
    import torch

    config = envelope or BallResidualEnvelope()
    if (
        raw_action.ndim != 2
        or raw_action.shape != previous.shape
        or raw_action.shape[1] != BALL_RESIDUAL_ACTION_SIZE
        or not 1 <= raw_action.shape[0] <= 4096
        or not bool(torch.isfinite(raw_action).all() and torch.isfinite(previous).all())
        or bool((previous.abs() > config.maximum_offset_rad).any())
    ):
        raise ValueError("finite bounded full-body residual state required")
    desired = config.maximum_offset_rad * torch.tanh(raw_action)
    return previous + torch.clamp(
        config.smoothing * (desired - previous),
        -config.maximum_step_rad,
        config.maximum_step_rad,
    )


def precontact_approach_reward(
    previous_distance: Any, distance: Any, already_contacted: Any
) -> Any:
    """Stop approach shaping after the first physical contact.

    Continuing to penalize ball-foot separation after a kick teaches the
    actor to retain the ball instead of launching it. The caller must derive
    ``already_contacted`` from physics, not geometric proximity or intent.
    """
    import torch

    if (
        previous_distance.ndim != 1
        or previous_distance.shape != distance.shape
        or distance.shape != already_contacted.shape
        or already_contacted.dtype != torch.bool
        or not bool(torch.isfinite(previous_distance).all() and torch.isfinite(distance).all())
        or bool((previous_distance < 0).any() or (distance < 0).any())
    ):
        raise ValueError("finite nonnegative distances and physical contact mask required")
    return 4 * (previous_distance - distance).clamp(-0.2, 0.2) * (~already_contacted).float()
