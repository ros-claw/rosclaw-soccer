"""Read-only contact chronology; later ball speed is not causal kick attribution."""

import math

import numpy as np


def contact_timing_metrics(
    force: np.ndarray, speed: np.ndarray, *, intended_foot: int, timestep: float
) -> dict[str, object]:
    """Summarize aligned post-step samples, with left/right columns and a 1 N gate."""
    if (
        type(intended_foot) is not int
        or intended_foot not in (0, 1)
        or not math.isfinite(timestep)
        or timestep <= 0
        or not isinstance(force, np.ndarray)
        or not isinstance(speed, np.ndarray)
        or force.ndim != 2
        or force.shape[1] != 2
        or len(force) == 0
        or speed.shape != (len(force),)
        or force.dtype.kind != "f"
        or speed.dtype.kind != "f"
        or not np.isfinite(force).all()
        or not np.isfinite(speed).all()
        or (force < 0).any()
        or (speed < 0).any()
    ):
        raise ValueError("finite aligned nonnegative post-step forces/speeds required")
    first = []
    for side in range(2):
        indices = np.flatnonzero(force[:, side] > 1.0)
        first.append(int(indices[0]) if len(indices) else None)
    intended = first[intended_foot]
    other = first[1 - intended_foot]
    touched = [i for i in first if i is not None]
    earliest = min(touched) if touched else None
    first_side = (
        "none"
        if earliest is None
        else "simultaneous"
        if first[0] == first[1]
        else "left"
        if first[0] == earliest
        else "right"
    )
    return dict(
        first_foot_contact_side=first_side,
        first_left_contact_time_s=None if first[0] is None else (first[0] + 1) * timestep,
        first_right_contact_time_s=None if first[1] is None else (first[1] + 1) * timestep,
        other_foot_contact_before_intended=bool(
            other is not None and (intended is None or other < intended)
        ),
        maximum_ball_speed_after_intended_contact_mps=(
            None if intended is None else float(speed[intended:].max())
        ),
        contact_chronology_proves_causal_kick=False,
    )
