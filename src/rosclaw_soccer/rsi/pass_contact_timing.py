"""Diagnostic-only timing of measured ball contacts in a recorded pass.

This does not infer safety or change a controller. Thresholds match the
development capture grade and the time base is supplied from the native model.
"""

import numpy as np


def contact_timing(forces: np.ndarray, timestep_s: float) -> dict:
    if (
        forces.ndim != 4
        or forces.shape[2:] != (2, 3)
        or not forces.shape[0]
        or not forces.shape[1]
        or not np.isfinite(forces).all()
        or (forces < 0).any()
        or not np.isfinite(timestep_s)
        or timestep_s <= 0
    ):
        raise ValueError("finite native [control, substep, agent, part] forces required")
    receiver = forces.reshape(-1, 2, 3)[:, 1, :]
    foot = np.flatnonzero(np.max(receiver[:, :2], axis=1) > 10.0)
    other = np.flatnonzero(receiver[:, 2] > 1.0)
    first_foot = int(foot[0]) if foot.size else None
    first_other = int(other[0]) if other.size else None
    return {
        "first_foot_substep": first_foot,
        "first_nonfoot_substep": first_other,
        "foot_first": first_foot is not None
        and first_other is not None
        and first_foot < first_other,
        "delay_s": (first_other - first_foot) * timestep_s
        if first_foot is not None and first_other is not None and first_foot < first_other
        else None,
    }
