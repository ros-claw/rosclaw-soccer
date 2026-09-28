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


def recovered_phase_offset_frames(
    frame: int,
    target: float,
    *,
    first_foot_contact_frame: int | None,
    recovery_frames: int,
) -> float:
    """Causally unwind a bounded phase lead after a foot-only first contact."""
    offset = phase_offset_frames(frame, target)
    if (
        type(recovery_frames) is not int
        or recovery_frames not in (12, 20, 30)
        or (
            first_foot_contact_frame is not None
            and (
                type(first_foot_contact_frame) is not int
                or not 0 <= first_foot_contact_frame < frame
            )
        )
    ):
        raise ValueError("invalid causal phase recovery")
    if first_foot_contact_frame is None:
        return offset
    return offset * max(0.0, 1.0 - (frame - first_foot_contact_frame) / recovery_frames)
