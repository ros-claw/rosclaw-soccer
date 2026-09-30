"""Metrics for auditable independent SIM_ONLY first-touch courses."""

from __future__ import annotations

from typing import Any

import numpy as np

FRAMES = 300


def post_contact_displacement(
    ball_position_m: np.ndarray[Any, Any], first_contact_frame: int | None
) -> dict[str, float | None]:
    """Measure actual ball displacement at a fixed horizon, never an intent."""
    ball = np.asarray(ball_position_m)
    if ball.shape != (FRAMES, 3) or not np.isfinite(ball).all():
        raise ValueError("finite 300-frame ball trajectory required")
    if first_contact_frame is None:
        return {"forward_60_m": None, "lateral_60_m": None, "lateral_over_forward_60": None}
    if type(first_contact_frame) is not int or not 0 <= first_contact_frame < FRAMES:
        raise ValueError("invalid first-contact frame")
    if first_contact_frame + 60 >= FRAMES:
        return {"forward_60_m": None, "lateral_60_m": None, "lateral_over_forward_60": None}
    delta = ball[first_contact_frame + 60] - ball[first_contact_frame]
    forward, lateral = float(delta[0]), float(delta[1])
    return {
        "forward_60_m": forward,
        "lateral_60_m": lateral,
        "lateral_over_forward_60": abs(lateral) / max(forward, 0.01),
    }
