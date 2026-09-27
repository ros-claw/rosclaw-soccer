"""Geometry-only virtual goal evidence for audited ball trajectories.

This does not create a physical goal, posts, crossbar, or net collision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class VirtualGoalCrossing:
    frame_after: int
    center_y_m: float
    center_z_m: float
    whole_ball_inside: bool


def first_virtual_goal_crossing(
    ball_position_m: NDArray[np.float64],
    *,
    goal_line_x_m: float = 5.0,
    goal_half_width_m: float = 1.2,
    goal_height_m: float = 1.6,
    ball_radius_m: float = 0.11,
) -> VirtualGoalCrossing | None:
    """Interpolate the first whole-ball crossing of a goal-line plane."""
    ball = np.asarray(ball_position_m, dtype=np.float64)
    if (
        ball.ndim != 2
        or ball.shape[0] < 2
        or ball.shape[1] != 3
        or not np.isfinite(ball).all()
        or not all(
            np.isfinite(value) and value > 0
            for value in (goal_line_x_m, goal_half_width_m, goal_height_m, ball_radius_m)
        )
        or ball_radius_m >= goal_half_width_m
        or ball_radius_m >= goal_height_m
    ):
        raise ValueError("finite physical ball trajectory and goal dimensions required")
    complete_x = goal_line_x_m + ball_radius_m
    indices = np.flatnonzero((ball[:-1, 0] < complete_x) & (ball[1:, 0] >= complete_x))
    if not indices.size:
        return None
    before = int(indices[0])
    start, end = ball[before], ball[before + 1]
    alpha = (complete_x - start[0]) / (end[0] - start[0])
    y, z = start[1:] + alpha * (end[1:] - start[1:])
    return VirtualGoalCrossing(
        frame_after=before + 1,
        center_y_m=float(y),
        center_z_m=float(z),
        whole_ball_inside=bool(
            abs(y) + ball_radius_m <= goal_half_width_m
            and z + ball_radius_m <= goal_height_m
            and z >= ball_radius_m - 0.005
        ),
    )
