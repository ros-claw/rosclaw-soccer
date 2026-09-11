"""Pure target-conditioned features and bounded handoff action shaping.

This module deliberately has no simulator, command, or authority access.  It is
the stable interface used by offline/online learners to condition a handoff on
the next outlet target instead of hiding the target in a reward oracle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HandoffPolicyConfig:
    max_yaw_rate: float = 1.2
    max_lateral_rate: float = 0.35
    max_forward_rate: float = 0.45


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
    """Return bounded egocentric handoff features for a learner."""
    _finite(target_dx, target_dy, body_yaw, ball_x, ball_y, residual_norm)
    target_heading = math.atan2(target_dy, target_dx)
    yaw_error = math.atan2(math.sin(target_heading - body_yaw), math.cos(target_heading - body_yaw))
    return (
        math.tanh(target_dx),
        math.tanh(target_dy),
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
    """Bounded interpretable fallback used while a learned policy is gated."""
    _finite(yaw_error, ball_x, ball_y)
    if config is None:
        config = HandoffPolicyConfig()
    if config.max_yaw_rate <= 0 or config.max_lateral_rate <= 0 or config.max_forward_rate <= 0:
        raise ValueError("handoff limits must be positive")
    yaw = max(-config.max_yaw_rate, min(config.max_yaw_rate, yaw_error))
    lateral = max(-config.max_lateral_rate, min(config.max_lateral_rate, -ball_y))
    forward = max(-config.max_forward_rate, min(config.max_forward_rate, 0.5 * ball_x))
    return yaw, forward, lateral
