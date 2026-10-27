"""Causal SIM-only post-contact velocity matching research teacher.

Only a bounded velocity command is proposed to the existing single neural
motor. The teacher has no ball, joint, clock, world, or hardware authority.
"""

import numpy as np


def post_touch_follow(
    baseline: np.ndarray,
    *,
    root_xy: np.ndarray,
    yaw: float,
    ball_xy: np.ndarray,
    ball_velocity_xy: np.ndarray,
    pass_direction: np.ndarray,
    contact_seen: bool,
    gain: float,
) -> np.ndarray:
    arrays = (baseline, root_xy, ball_xy, ball_velocity_xy, pass_direction)
    if (
        baseline.shape != (3,)
        or any(x.shape != (2,) for x in arrays[1:])
        or any(not np.isfinite(x).all() for x in arrays)
        or not np.isfinite(yaw)
        or type(contact_seen) is not bool
        or type(gain) is not float
        or gain not in (0.0, 0.25, 0.5, 0.75)
        or abs(np.linalg.norm(pass_direction) - 1) > 1e-10
        or not (-0.5 <= baseline[0] <= 1 and -0.5 <= baseline[1] <= 0.5 and -1 <= baseline[2] <= 1)
    ):
        raise ValueError("bounded measured own-state post-contact command required")
    if gain == 0 or not contact_seen or np.linalg.norm(ball_xy - root_xy) > 0.65:
        return baseline.copy()
    target_root = ball_xy - 0.30 * pass_direction
    world_velocity = ball_velocity_xy + 1.2 * (target_root - root_xy)
    magnitude = float(np.linalg.norm(world_velocity))
    if magnitude > 1:
        world_velocity /= magnitude
    c, s = np.cos(yaw), np.sin(yaw)
    desired = np.array(
        [
            c * world_velocity[0] + s * world_velocity[1],
            -s * world_velocity[0] + c * world_velocity[1],
        ]
    )
    result = baseline.astype(np.float64, copy=True)
    result[:2] = (1 - gain) * result[:2] + gain * desired
    result[0] = np.clip(result[0], -0.5, 1)
    result[1] = np.clip(result[1], -0.5, 0.5)
    return result.astype(baseline.dtype)
