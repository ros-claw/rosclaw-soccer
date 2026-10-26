"""Measured SIM stage tracking diagnostics, never a controller or safety gate."""

import math

import numpy as np


def signed_stage_tracking(
    positions: np.ndarray,
    velocities: np.ndarray,
    *,
    intended_displacement: np.ndarray,
    native_timestep_s: float,
) -> dict[str, float | int | bool | None]:
    """Describe signed measured movement along a declared horizontal waypoint.

    Positions contain the initial observation followed by each completed native
    step; velocities align with those positions. Do not pad censored traces.
    """
    if (
        type(positions) is not np.ndarray
        or type(velocities) is not np.ndarray
        or positions.dtype != np.float64
        or velocities.dtype != np.float64
        or positions.ndim != 2
        or positions.shape[1] != 2
        or velocities.shape != positions.shape
        or len(positions) < 2
        or not np.isfinite(positions).all()
        or not np.isfinite(velocities).all()
        or type(intended_displacement) is not np.ndarray
        or intended_displacement.dtype != np.float64
        or intended_displacement.shape != (2,)
        or not np.isfinite(intended_displacement).all()
        or type(native_timestep_s) not in (float, int)
        or not math.isfinite(native_timestep_s)
        or native_timestep_s <= 0
    ):
        raise ValueError("finite aligned measured stage trajectory required")
    distance = float(np.linalg.norm(intended_displacement))
    if distance <= 1e-9:
        raise ValueError("nonzero declared intended displacement required")
    direction = intended_displacement / distance
    progress = (positions - positions[0]) @ direction
    speed = velocities @ direction
    positive = np.flatnonzero(progress[1:] > 0)
    return dict(
        native_steps=len(positions) - 1,
        observed_duration_s=(len(positions) - 1) * native_timestep_s,
        requested_displacement_m=distance,
        measured_signed_displacement_m=float(progress[-1]),
        maximum_opposite_displacement_m=max(0.0, float(-np.min(progress))),
        final_signed_velocity_mps=float(speed[-1]),
        first_positive_progress_s=(
            None if not len(positive) else float((positive[0] + 1) * native_timestep_s)
        ),
        physical_safety_or_task_success_inferred=False,
    )
