"""Sampled goal-plane geometry for training feedback, not goal attribution.

The caller must supply an independently identified shot frame.  This helper
only interpolates the first subsequent outward crossing of the declared goal
plane.  It neither proves who caused the crossing nor asserts an official goal,
and it deliberately returns no fabricated position for a ball that never
reaches the plane.  Later touches, posts, goalkeeper contacts and any motion
between stored frames remain outside this observation's evidence boundary.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


def observe_shot_mouth_crossing(
    trajectory: dict[str, NDArray[Any]],
    *,
    shot_frame: int,
    goal_side: Literal["left", "right"],
    target_yz_m: tuple[float, float],
) -> dict[str, Any]:
    """Measure a post-shot crossing against the physical ball-center aperture.

    The first outward center-plane crossing is interpolated between adjacent
    recorded samples; no simulated future, contact causality, or outcome label
    is inferred. The target must be inside the whole-ball-clearance aperture.
    """
    if type(shot_frame) is not int or shot_frame < 0 or goal_side not in ("left", "right"):
        raise ValueError("nonnegative shot frame and explicit goal side required")
    if (
        type(target_yz_m) is not tuple
        or len(target_yz_m) != 2
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in target_yz_m)
    ):
        raise ValueError("finite goal target required")
    time = np.asarray(trajectory.get("time", ()), dtype=np.float64)
    ball = np.asarray(trajectory.get("ball_pose", ()), dtype=np.float64)
    geometry = np.asarray(trajectory.get("pitch_boundary_geometry", ()), dtype=np.float64)
    count = len(time) if time.ndim == 1 else 0
    if (
        count < 2
        or ball.shape != (count, 7)
        or geometry.shape != (count, 6)
        or shot_frame >= count
        or not np.isfinite(time).all()
        or not np.isfinite(ball).all()
        or not np.isfinite(geometry).all()
        or np.any(np.diff(time) <= 0)
        or not np.array_equal(geometry, np.broadcast_to(geometry[0], geometry.shape))
    ):
        raise ValueError("finite aligned trajectory and constant geometry required")
    left, right, half_pitch_width, radius, width, height = geometry[0]
    target_y, target_z = target_yz_m
    if (
        not left < right
        or half_pitch_width <= 0
        or radius <= 0
        or width <= 2 * radius
        or height <= 2 * radius
        or abs(target_y) > width / 2 - radius
        or not radius <= target_z <= height - radius
    ):
        raise ValueError("valid field, ball, goal and whole-ball target required")
    plane = left if goal_side == "left" else right
    direction = -1.0 if goal_side == "left" else 1.0
    # A sampled shot that starts on or behind the goal plane cannot identify a
    # later crossing without inventing an earlier in-field ball state.
    if direction * (ball[shot_frame, 0] - plane) >= 0:
        raise ValueError("shot must begin inside the selected goal plane")
    result: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.shot_mouth_observation.v1",
        "goal_side": goal_side,
        "shot_frame": shot_frame,
        "target_yz_m": [float(target_y), float(target_z)],
        "crossed": False,
        "crossing_frame_before": None,
        "crossing_time_sec": None,
        "crossing_center_yz_m": None,
        "target_error_m": None,
        "whole_ball_lateral_clearance_m": None,
        "whole_ball_vertical_clearance_m": None,
        "center_crossing_fits_mouth": None,
        "goal_scored": None,
        "hardware_authorized": False,
    }
    for before in range(shot_frame, count - 1):
        previous = direction * (ball[before, 0] - plane)
        following = direction * (ball[before + 1, 0] - plane)
        if previous < 0 <= following:
            alpha = float(-previous / (following - previous))
            y, z = ball[before, 1:3] + alpha * (ball[before + 1, 1:3] - ball[before, 1:3])
            lateral = float(width / 2 - radius - abs(y))
            vertical = float(min(z - radius, height - radius - z))
            result.update(
                crossed=True,
                crossing_frame_before=before,
                crossing_time_sec=float(time[before] + alpha * (time[before + 1] - time[before])),
                crossing_center_yz_m=[float(y), float(z)],
                target_error_m=math.hypot(float(y - target_y), float(z - target_z)),
                whole_ball_lateral_clearance_m=lateral,
                whole_ball_vertical_clearance_m=vertical,
                center_crossing_fits_mouth=lateral >= 0 and vertical >= 0,
            )
            break
    return result
