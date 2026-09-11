"""Observable receiving-phase cue, not skill admission or motion authority.

An experimental preparatory learner may use phase zero before a nearby
approaching ball is observed. The clock then advances without restarting when
the ball recedes. Callers own per-player state, episode boundaries and physical
admission; recorded future entry times are not inputs to this interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReceptionPhase:
    start_frame: int | None
    fraction: float


def advance_reception_phase(
    *,
    frame: int,
    start_frame: int | None,
    ball_offset_xy: tuple[float, float],
    relative_velocity_xy: tuple[float, float],
    approach_radius_m: float = 1.2,
    phase_frames: int = 100,
) -> ReceptionPhase:
    """Latch the first nearby approaching-ball observation; cap at (N-1)/N.

    Offsets and relative velocities must use the same world or body frame.
    Missing frames do not reset time. No ball-contact or readiness claim is
    inferred from this kinematic cue. Passing None starts a new caller-owned
    clock and must not be used to silently restart an active skill.
    """
    if (
        type(frame) is not int
        or not 0 <= frame <= 1_000_000_000
        or (
            start_frame is not None
            and (type(start_frame) is not int or not 0 <= start_frame <= frame)
        )
        or type(phase_frames) is not int
        or not 1 <= phase_frames <= 10000
        or type(approach_radius_m) not in (int, float)
        or not math.isfinite(approach_radius_m)
        or not 0.1 <= approach_radius_m <= 2.0
    ):
        raise ValueError("bounded receiving phase clock required")
    for pair in (ball_offset_xy, relative_velocity_xy):
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(
                type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 100 for x in pair
            )
        ):
            raise ValueError("finite bounded planar receiving state required")
    dot = sum(p * v for p, v in zip(ball_offset_xy, relative_velocity_xy, strict=True))
    if start_frame is None and math.hypot(*ball_offset_xy) <= approach_radius_m and dot < 0:
        start_frame = frame
    fraction = (
        0.0 if start_frame is None else min(frame - start_frame, phase_frames - 1) / phase_frames
    )
    return ReceptionPhase(start_frame, fraction)
