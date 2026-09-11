"""Pure target-conditioned features and bounded handoff action shaping.

This experimental feature/proposal interface has no simulator or authority
access. It is not a trained policy and has not passed physical handoff exams.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HandoffPolicyConfig:
    max_yaw_rate: float = 1.2
    max_lateral_rate: float = 0.35
    max_forward_rate: float = 0.45

    def __post_init__(self) -> None:
        limits = (self.max_yaw_rate, self.max_lateral_rate, self.max_forward_rate)
        _finite(*limits)
        if any(v <= 0 for v in limits):
            raise ValueError("handoff limits must be positive")


def _finite(*values: float) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ValueError("handoff inputs must be finite")


def build_handoff_features(
    *,
    target_dx: float,
    target_dy: float,
    body_yaw: float,
    ball_x: float,
    ball_y: float,
    residual_norm: float,
) -> tuple[float, ...]:
    """World target displacement, body-frame ball position, scalar residual size.

    The scalar residual norm does not encode joint or recurrent history. A
    learner must receive that history separately. Coincident targets have no
    defined heading and are rejected.
    """
    _finite(target_dx, target_dy, body_yaw, ball_x, ball_y, residual_norm)
    if residual_norm < 0 or math.hypot(target_dx, target_dy) < 1e-9:
        raise ValueError("nonnegative residual norm and nonzero target required")
    c, s = math.cos(body_yaw), math.sin(body_yaw)
    local_x, local_y = c * target_dx + s * target_dy, -s * target_dx + c * target_dy
    _finite(local_x, local_y)
    target_heading = math.atan2(target_dy, target_dx)
    yaw_error = math.atan2(math.sin(target_heading - body_yaw), math.cos(target_heading - body_yaw))
    return (
        math.tanh(local_x),
        math.tanh(local_y),
        math.sin(yaw_error),
        math.cos(yaw_error),
        math.tanh(ball_x),
        math.tanh(ball_y),
        math.tanh(residual_norm),
    )


def shape_handoff_action(
    *,
    yaw_error: float,
    ball_x: float,
    ball_y: float,
    config: HandoffPolicyConfig | None = None,
) -> tuple[float, float, float]:
    """Experimental proposal (yaw rate, forward speed, lateral speed).

    Clipping bounds magnitudes only; it guarantees neither balance nor action
    continuity. There is no automatic runtime fallback or actuation here.
    """
    _finite(yaw_error, ball_x, ball_y)
    if config is None:
        config = HandoffPolicyConfig()
    yaw = max(-config.max_yaw_rate, min(config.max_yaw_rate, yaw_error))
    lateral = max(-config.max_lateral_rate, min(config.max_lateral_rate, -ball_y))
    forward = max(-config.max_forward_rate, min(config.max_forward_rate, 0.5 * ball_x))
    return yaw, forward, lateral


def build_handoff_actor_critic() -> Any:
    """Optional Torch learner: 133 motor + 3 navigation history + 7 goal features.

    The motor block includes all 29 previous joint residuals. Navigation is
    selected before advancing the frozen motor foundation, so this block must
    use the previous foundation target. Outputs are raw navigation proposals,
    requiring bounded composition and scene clearance in a simulation caller.
    """
    import torch

    from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic

    class HandoffActorCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            parent = build_ball_residual_actor_critic()
            self.actor, self.critic = parent.actor, parent.critic
            self.actor[0] = torch.nn.Linear(143, 128)
            self.critic[0] = torch.nn.Linear(143, 128)
            self.actor[-1] = torch.nn.Linear(128, 3)
            torch.nn.init.zeros_(self.actor[-1].weight)
            torch.nn.init.zeros_(self.actor[-1].bias)
            self.logstd = torch.nn.Parameter(torch.full((3,), -5.3))

        def forward(self, observation: Any) -> tuple[Any, Any]:
            if (
                not isinstance(observation, torch.Tensor)
                or observation.ndim != 2
                or observation.shape[1] != 143
                or not 1 <= len(observation) <= 65536
                or observation.dtype != torch.float32
                or observation.layout != torch.strided
                or not bool(torch.isfinite(observation).all())
                or bool((observation.abs() > 10).any())
                or bool((observation[:, 133:].abs() > 1.000001).any())
            ):
                raise ValueError("finite 143-feature handoff state with bounded history required")
            mean, value = self.actor(observation), self.critic(observation).squeeze(-1)
            if not bool(torch.isfinite(mean).all() and torch.isfinite(value).all()):
                raise FloatingPointError("nonfinite handoff actor or critic output")
            return mean, value

    return HandoffActorCritic()
