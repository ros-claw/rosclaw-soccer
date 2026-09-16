"""Causal goal-relative positioning proposals, not motor or save authority."""

from __future__ import annotations

import math

import numpy as np


def goalkeeper_cover_target(
    *,
    ball_position_m: tuple[float, float, float],
    ball_velocity_mps: tuple[float, float, float],
    own_goal_m: tuple[float, float, float],
    opponent_goal_m: tuple[float, float, float],
    depth_m: float,
    lateral_limit_m: float = 1.25,
    interception_horizon_sec: float = 1.5,
) -> tuple[float, float, float]:
    """Cover the goal-centre/ball ray; intercept a measured incoming trajectory.

    A distant stationary ball must not pull the keeper sideways by its full
    lateral displacement. Project that ray onto the keeper's depth plane. For
    an incoming ball reaching this plane within the bounded horizon, use its
    measured planar velocity instead. No future simulator state is consulted.
    Existing navigation, collision and motor limits still own execution.
    """
    vectors = []
    for value in (ball_position_m, ball_velocity_mps, own_goal_m, opponent_goal_m):
        array = np.asarray(value)
        if (
            array.shape != (3,)
            or array.dtype.kind not in "fiu"
            or not np.isfinite(array).all()
            or np.any(np.abs(array) > 10000)
        ):
            raise ValueError("finite bounded measured positions and velocity required")
        vectors.append(array.astype(np.float64))
    if (
        any(
            type(v) not in (float, int) or not math.isfinite(v)
            for v in (depth_m, lateral_limit_m, interception_horizon_sec)
        )
        or not 0.20 <= depth_m <= 1.55
        or not 0.10 <= lateral_limit_m <= 5.0
        or not 0.05 <= interception_horizon_sec <= 2.0
    ):
        raise ValueError("bounded explicit goal-cover envelope required")
    ball, velocity, own, opponent = vectors
    axis = opponent[:2] - own[:2]
    length = float(np.linalg.norm(axis))
    if not 1.0 <= length <= 200:
        raise ValueError("distinct bounded goal centres required")
    axis /= length
    lateral_axis = np.array((-axis[1], axis[0]))
    relative = ball[:2] - own[:2]
    forward = float(relative @ axis)
    lateral = float(relative @ lateral_axis)
    target_lateral = lateral * depth_m / max(depth_m, forward)
    forward_velocity = float(velocity[:2] @ axis)
    if forward_velocity < -0.10:
        arrival = (depth_m - forward) / forward_velocity
        if 0 < arrival <= interception_horizon_sec:
            target_lateral = lateral + arrival * float(velocity[:2] @ lateral_axis)
    target = (
        own[:2]
        + depth_m * axis
        + float(np.clip(target_lateral, -lateral_limit_m, lateral_limit_m)) * lateral_axis
    )
    return float(target[0]), float(target[1]), 0.0
