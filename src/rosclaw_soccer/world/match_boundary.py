"""Whole-ball exit observations for a declared small-sided training pitch."""

from __future__ import annotations

import math
from collections.abc import Sequence


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
