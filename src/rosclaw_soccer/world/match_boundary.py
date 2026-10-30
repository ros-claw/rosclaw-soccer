"""Whole-ball exit observations for a declared small-sided training pitch."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def require_in_play_ball_launch(
    position: Sequence[float],
    *,
    left_x: float,
    right_x: float,
    radius: float,
    half_width: float = 3.0,
) -> None:
    """Require a whole ball initially inside the declared small-sided pitch.

    This is a conservative match-exam input check, not physics or safety
    qualification. It intentionally does not alter general-purpose simulator
    scenarios such as out-of-play recovery and does not move the ball.
    """
    try:
        if len(position) != 3:
            raise ValueError("three-dimensional ball launch required")
        x, y, z = (float(v) for v in position)
        geometry = (float(left_x), float(right_x), float(radius), float(half_width))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("finite match launch and geometry required") from exc
    left, right, ball_radius, width = geometry
    if (
        not all(math.isfinite(v) for v in (x, y, z, *geometry))
        or ball_radius <= 0
        or width <= ball_radius
        or right - left <= 2 * ball_radius
        or not left + ball_radius <= x <= right - ball_radius
        or abs(y) > width - ball_radius
        or z < ball_radius - 0.005
    ):
        raise ValueError("whole ball must start inside the declared match pitch")


def ball_exit_reason(
    position: Sequence[float],
    *,
    left_x: float,
    right_x: float,
    radius: float,
    goal_width: float,
    goal_height: float,
) -> str | None:
    if len(position) != 3 or not left_x < right_x or radius <= 0:
        raise ValueError("invalid match boundary geometry or ball position")
    if any(
        not math.isfinite(v) for v in (*position, left_x, right_x, radius, goal_width, goal_height)
    ):
        return "NONFINITE_STATE"
    x, y, z = position
    if abs(y) > 3.0 + radius:
        return "TOUCHLINE_OUT"
    if x < left_x - radius or x > right_x + radius:
        mouth = abs(y) <= goal_width / 2.0 - radius and radius <= z <= goal_height - radius
        return (
            ("BLUE_GOAL_CROSSING" if x < left_x else "RED_GOAL_CROSSING")
            if mouth
            else "GOAL_LINE_OUT"
        )
    return None


def uninterrupted_in_play(trajectory: dict[str, NDArray[Any]]) -> dict[str, Any]:
    """Describe sampled ball exposure; this does not invent a referee restart.

    An exit on the last recorded frame is terminal, not continuous play. A
    future restart-aware exam needs an independently verified restart contract.
    """
    time = np.asarray(trajectory.get("time", ()), dtype=np.float64)
    ball = np.asarray(trajectory.get("ball_pose", ()), dtype=np.float64)
    geometry = np.asarray(trajectory.get("pitch_boundary_geometry", ()), dtype=np.float64)
    count = len(time) if time.ndim == 1 else 0
    if (
        count < 1
        or ball.shape != (count, 7)
        or geometry.shape != (count, 6)
        or not np.isfinite(time).all()
        or not np.isfinite(ball).all()
        or not np.isfinite(geometry).all()
        or np.any(np.diff(time) <= 0)
        or not np.array_equal(geometry, np.broadcast_to(geometry[0], geometry.shape))
    ):
        raise ValueError("finite aligned constant match boundary evidence required")
    left, right, half_width, radius, width, height = geometry[0]
    if not (left < right and half_width == 3.0 and 0 < radius < min(width, height) / 2):
        raise ValueError("unsupported match boundary geometry")
    for frame in range(count):
        reason = ball_exit_reason(
            ball[frame, :3].tolist(),
            left_x=left,
            right_x=right,
            radius=radius,
            goal_width=width,
            goal_height=height,
        )
        if reason is not None:
            return {
                "schema_version": "rosclaw_soccer.uninterrupted_in_play.v1",
                "uninterrupted": False,
                "first_exit_frame": frame,
                "first_exit_time_sec": float(time[frame]),
                "first_exit_reason": reason,
            }
    return {
        "schema_version": "rosclaw_soccer.uninterrupted_in_play.v1",
        "uninterrupted": True,
        "first_exit_frame": None,
        "first_exit_time_sec": None,
        "first_exit_reason": None,
    }
