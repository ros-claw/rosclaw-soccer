"""Exact measured approach command shared by actual and prospective SIM bodies.

Preserves the legacy float32 command slew and cached MuJoCo world geometry.
No neural inference, physics step, parameter search or new action authority.
"""

from typing import Any

import numpy as np


def approach_command(
    *,
    q: np.ndarray,
    v: np.ndarray,
    rotation: np.ndarray,
    cached_ball_position: np.ndarray,
    previous: np.ndarray,
    parameters: dict[str, Any],
    fast: bool,
    feedback: bool,
    halt: bool = False,
) -> np.ndarray:
    if any(type(flag) is not bool for flag in (fast, feedback, halt)) or any(
        type(value) is not np.ndarray
        or value.shape != shape
        or value.dtype != dtype
        or not np.isfinite(value).all()
        for value, shape, dtype in (
            (q, (43,), np.float64),
            (v, (41,), np.float64),
            (rotation, (3, 3), np.float64),
            (cached_ball_position, (3,), np.float64),
            (previous, (3,), np.float32),
        )
    ):
        raise ValueError("typed finite measured body, cached geometry and prior command required")
    desired_world = np.array(
        [
            parameters["speed_mps"],
            np.clip(
                2 * (cached_ball_position[1] + parameters["root_lateral_offset_m"] - q[1]),
                -0.4,
                0.4,
            ),
        ]
    )
    if feedback:
        from rosclaw_soccer.rsi.feedback_kick_approach import velocity_matched_approach

        desired_world = velocity_matched_approach(
            root_position=q[:3],
            ball_position=q[36:39],
            ball_velocity=v[35:38],
            cruise_speed=parameters["speed_mps"],
            entry_gap=parameters["entry_gap_m"],
            lateral_offset=parameters["root_lateral_offset_m"],
        )
    if halt:
        desired_world[:] = 0.0
    xy = rotation[:2, :2].T @ desired_world
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    wanted = np.array(
        [
            np.clip(xy[0], -0.5, 2.0 if fast else 1.0),
            np.clip(xy[1], -0.5, 0.5),
            np.clip(-2 * yaw, -1, 1),
        ],
        np.float32,
    )
    if halt:
        wanted[:] = 0.0
    if fast:
        norm = float(np.linalg.norm(wanted[:2]))
        if norm > 2:
            wanted[:2] *= np.float32(2 / norm)
    result: np.ndarray = previous + np.clip(wanted - previous, -0.05, 0.05)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite approach command; no implicit repair")
    return result
