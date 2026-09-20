"""Causal future-knot proposals for a privileged, simulated contact teacher.

Interpolation couples neighboring knots: freezing only knots before the query
time would silently change the already executed past. Physics prefix replay is
still required by the caller. This helper alone is not MPC or a learned actor.
"""

from dataclasses import replace

import numpy as np

from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def locked_knot_count(schedule: ReceivingOracleSchedule, *, branch_frame: int) -> int:
    """How many knots influence control frames strictly before this branch?"""
    schedule.__post_init__()
    if type(branch_frame) is not int or not 0 <= branch_frame <= 1000:
        raise ValueError("bounded integer branch control frame required")
    if branch_frame <= schedule.start_frame:
        return 0
    last_executed_offset = branch_frame - 1 - schedule.start_frame
    last_influential_knot = (
        last_executed_offset + schedule.knot_frames - 1
    ) // schedule.knot_frames
    return min(len(schedule.knots), last_influential_knot + 1)


def propose_continuations(
    schedule: ReceivingOracleSchedule,
    *,
    branch_frame: int,
    seed: int,
    count: int = 4,
    standard_deviation: float = 0.15,
) -> tuple[ReceivingOracleSchedule, ...]:
    """Return incumbent plus bounded future-only perturbations.

    Scores and actual branch-state commitments come from the external CPU
    simulator, never from this proposal function. No success gate is relaxed.
    """
    locked = locked_knot_count(schedule, branch_frame=branch_frame)
    if (
        type(seed) is not int
        or not 0 <= seed < 2**32
        or type(count) is not int
        or not 1 <= count <= 64
        or type(standard_deviation) not in (int, float)
        or not np.isfinite(standard_deviation)
        or not 0 < standard_deviation <= 0.5
    ):
        raise ValueError("bounded fixed search budget and seed required")
    if locked == len(schedule.knots):
        raise ValueError("no unexecuted knots remain; cannot rewrite held action history")
    rng = np.random.default_rng(seed)
    source = np.asarray(schedule.knots, dtype=np.float64)
    proposals = [schedule]
    for _ in range(count):
        values = source.copy()
        values[locked:] = np.clip(
            values[locked:] + rng.normal(0, standard_deviation, values[locked:].shape), -1, 1
        )
        proposals.append(
            replace(schedule, knots=tuple(tuple(float(v) for v in row) for row in values))
        )
    return tuple(proposals)


def timing_continuations(
    schedule: ReceivingOracleSchedule, *, branch_frame: int
) -> tuple[ReceivingOracleSchedule, ...]:
    """Fixed timing/amplitude probes guided by the teacher-seeding diagnosis.

    Future reference shifts are -200, -100, +100, +200 ms at 50 Hz;
    amplitude factors are 0.5 and 1.5. Executed/interpolating prefix knots
    remain immutable, all outputs remain in the original [-1, 1] envelope.
    These are proposals for physics evaluation, not automatic corrections.
    """
    locked = locked_knot_count(schedule, branch_frame=branch_frame)
    if locked == len(schedule.knots):
        raise ValueError("no unexecuted knots remain")
    source = np.asarray(schedule.knots, dtype=np.float64)
    times = np.arange(len(source)) * schedule.knot_frames
    variants = []
    for shift in (-10, -5, 5, 10):
        values = source.copy()
        for dimension in range(source.shape[1]):
            values[locked:, dimension] = np.interp(
                times[locked:] + shift, times, source[:, dimension]
            )
        variants.append(values)
    for factor in (0.5, 1.5):
        values = source.copy()
        values[locked:] = np.clip(source[locked:] * factor, -1, 1)
        variants.append(values)
    return (schedule,) + tuple(
        replace(schedule, knots=tuple(tuple(float(v) for v in row) for row in values))
        for values in variants
    )
