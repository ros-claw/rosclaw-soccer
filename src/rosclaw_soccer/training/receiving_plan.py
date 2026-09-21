"""Bounded time-varying leg plans for private receiving forecasts.

This samples an existing proposal, not a teacher or a success certificate.
The execution owner still applies admission, smoothing and rate limits. A
research library must authenticate its demonstrations separately; copying a
schedule does not make a seen trajectory independent evaluation evidence.
"""

import numpy as np

from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def receiving_desired_plan(
    schedule: ReceivingOracleSchedule,
    *,
    start_frame: int,
    control_frames: int,
    phase_shift_frames: int = 0,
) -> tuple[tuple[float, ...], ...]:
    """Sample consecutive 50 Hz desired offsets before the original filter.

    Positive phase shift advances the reference, not physical time. References
    preceding the schedule entry hold its first knot; references after its last
    knot hold the last. This explicit rule also applies between replans: callers
    must execute the corresponding row, not repeatedly execute the first row.
    No actual future state or future admission decision is consulted here.
    """
    schedule.__post_init__()
    if (
        schedule.substrate != "A0_leg12"
        or type(start_frame) is not int
        or not schedule.start_frame <= start_frame < 1000
        or type(control_frames) is not int
        or not 1 <= control_frames <= 100
        or start_frame + control_frames > 1000
        or type(phase_shift_frames) is not int
        or not -100 <= phase_shift_frames <= 100
    ):
        raise ValueError("bounded post-entry 12-joint plan required")
    knots = np.asarray(schedule.knots)
    rows = []
    for frame in range(start_frame, start_frame + control_frames):
        reference = max(schedule.start_frame, frame + phase_shift_frames)
        offset = (reference - schedule.start_frame) / schedule.knot_frames
        left = min(int(offset), len(knots) - 1)
        right = min(left + 1, len(knots) - 1)
        fraction = min(offset - left, 1.0)
        desired = 0.1 * ((1 - fraction) * knots[left] + fraction * knots[right])
        rows.append(tuple(float(value) for value in desired))
    return tuple(rows)
