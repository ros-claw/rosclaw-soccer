"""Recorded one-control-step physical labels, never policy observations.

The caller authenticates traces and courses. These targets neither prove
counterfactual effects nor grant execution authority. Positions follow the
source trace's MuJoCo body-position sampling convention, not an invented
fresh-forward kinematics convention.
"""

from typing import Any

import numpy as np


def recorded_transition_targets(trace: dict[str, Any]) -> dict[str, np.ndarray[Any, Any]]:
    if type(trace) is not dict:
        raise ValueError("complete single-episode measured trace required")
    shapes = {
        "ball_position_before_step_m": (300, 1, 3),
        "ball_position_after_step_m": (300, 1, 3),
        "force_n": (300, 1, 6),
        "pelvis_z_per_substep_m": (300, 1, 10),
        "motor_delta_rad": (300, 1, 12),
    }
    values = {}
    for key, shape in shapes.items():
        raw = np.asarray(trace.get(key))
        if (
            raw.shape != shape
            or raw.dtype.kind not in "fiu"
            or not np.isfinite(raw).all()
            or np.max(np.abs(raw)) > 1e6
        ):
            raise ValueError("complete finite measured transition arrays required")
        values[key] = np.array(raw, dtype=np.float64, copy=True)[:, 0]
    before, after = values["ball_position_before_step_m"], values["ball_position_after_step_m"]
    force, pelvis = values["force_n"], values["pelvis_z_per_substep_m"]
    if (
        not np.array_equal(after[:-1], before[1:])
        or np.any(force < 0)
        or np.any(pelvis < 0)
        or np.max(np.abs(values["motor_delta_rad"])) > 0.1600000001
    ):
        raise ValueError("aligned unchanged sampling and bounded actual action required")
    # Frames 30..299 match original learner rows. The final row has recorded
    # after-step positions, contact peaks and all ten pelvis samples too.
    return {
        "recorded_ball_displacement_m": (after - before)[30:].copy(),
        "actual_executed_motor_delta_rad": values["motor_delta_rad"][30:].copy(),
        "contact_force_peak_n": force[30:].copy(),
        "foot_contact": np.any(force[30:, :2] > 1, axis=1),
        "nonfoot_contact": np.any(force[30:, 2:] > 1, axis=1),
        "minimum_pelvis_z_m": pelvis[30:].min(axis=1),
        "pelvis_below_original_floor": np.any(pelvis[30:] < 0.65, axis=1),
    }
