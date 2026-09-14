"""Trace-based whole-ball goal entry, separate from shot-target precision.

This is observation evidence, not a save certificate: a finite trace without
a goal cannot prove that a ball will never enter later. Historical receipts
remain unchanged when this additional audit rejects a claimed save.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.world.field import G1TrainingGoalSpec, g1_ball_inside_goal_mouth


@dataclass(frozen=True)
class GoalEntryAudit:
    complete_forward_crossings: int
    goal_entries: int
    first_goal_time_sec: float | None
    first_goal_center_m: tuple[float, float, float] | None
    ground_contact_tolerance_m: float
    maximum_observed_sample_dt_sec: float
    whole_ball_plane_x_m: float
    ball_radius_m: float
    goal_width_m: float
    goal_height_m: float

    @property
    def goal_observed(self) -> bool:
        return self.goal_entries > 0


def audit_goal_entries(
    *,
    time_sec: NDArray[np.float64],
    ball_position_m: NDArray[np.float64],
    goal: G1TrainingGoalSpec,
    ground_contact_tolerance_m: float = 0.001,
    maximum_sample_dt_sec: float = 0.02,
) -> GoalEntryAudit:
    """Inspect every forward crossing of the trailing ball surface.

    Positive X is toward the goal. Crossing positions use linear interpolation
    between supplied samples; the reported cadence is part of the evidence.
    The aperture's existing bounded floor tolerance does not enlarge its posts
    or crossbar, and does not modify the separate shot-precision threshold.
    """
    times = np.asarray(time_sec, dtype=np.float64)
    positions = np.asarray(ball_position_m, dtype=np.float64)
    if (
        times.ndim != 1
        or len(times) < 2
        or positions.shape != (len(times), 3)
        or not np.isfinite(times).all()
        or not np.isfinite(positions).all()
        or times[0] < 0
        or isinstance(maximum_sample_dt_sec, bool)
        or not math.isfinite(maximum_sample_dt_sec)
        or not 0 < maximum_sample_dt_sec <= 0.02
    ):
        raise ValueError("invalid finite goal-entry trace or cadence bound")
    # Validate tolerance even when the trace contains no crossing.
    g1_ball_inside_goal_mouth(
        goal,
        ball_y_m=0,
        ball_z_m=goal.ball_radius_m,
        ground_contact_tolerance_m=ground_contact_tolerance_m,
    )
    try:
        with np.errstate(over="raise", invalid="raise"):
            dt = np.diff(times)
    except FloatingPointError as exc:
        raise ValueError("goal-entry clock arithmetic overflow") from exc
    if np.any(dt <= 0) or np.any(dt > maximum_sample_dt_sec + 1e-9):
        raise ValueError("goal-entry trace is unordered or exceeds the cadence bound")
    plane = goal.plane_x_m + goal.ball_radius_m
    if positions[0, 0] >= plane:
        raise ValueError("trace starts beyond the whole-ball goal plane")
    crossings = entries = 0
    first_time = None
    first_position = None
    for index in range(1, len(times)):
        previous, current = positions[index - 1 : index + 1]
        if not previous[0] < plane <= current[0]:
            continue
        crossings += 1
        try:
            with np.errstate(over="raise", invalid="raise"):
                fraction = float((plane - previous[0]) / (current[0] - previous[0]))
                center = previous + fraction * (current - previous)
        except FloatingPointError as exc:
            raise ValueError("goal-entry interpolation overflow") from exc
        if not g1_ball_inside_goal_mouth(
            goal,
            ball_y_m=float(center[1]),
            ball_z_m=float(center[2]),
            ground_contact_tolerance_m=ground_contact_tolerance_m,
        ):
            continue
        entries += 1
        if first_time is None:
            first_time = float(times[index - 1] + fraction * dt[index - 1])
            first_position = (float(center[0]), float(center[1]), float(center[2]))
    return GoalEntryAudit(
        crossings,
        entries,
        first_time,
        first_position,
        float(ground_contact_tolerance_m),
        float(dt.max()),
        plane,
        goal.ball_radius_m,
        goal.width_m,
        goal.height_m,
    )
