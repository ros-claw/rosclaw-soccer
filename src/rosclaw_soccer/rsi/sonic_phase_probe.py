"""Bounded, deterministic SIM_ONLY SONIC reference-phase experiment."""

from __future__ import annotations

import math


def phase_offset_frames(frame: int, target: float, *, ramp_frames: int = 20) -> float:
    """Ramp the reference clock without a discontinuity at snapshot restoration."""
    if (
        type(frame) is not int
        or frame < 0
        or type(target) is bool
        or not isinstance(target, (int, float))
        or not math.isfinite(target)
        or abs(target) > 6.0
        or type(ramp_frames) is not int
        or not 12 <= ramp_frames <= 40
        or abs(target) / ramp_frames > 0.5
    ):
        raise ValueError("phase probe exceeds SIM_ONLY bounds")
    return float(target) * min(frame / ramp_frames, 1.0)
