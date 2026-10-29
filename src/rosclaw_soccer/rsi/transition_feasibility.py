"""SIM-only geometry of two simultaneous joint-target constraints.

This is a diagnostic, not a controller, safety certificate or learned policy.
All inputs are joint targets in the same physical joint order and radians.
"""

import numpy as np


def transition_feasibility(
    previous_target: np.ndarray,
    reference_target: np.ndarray,
    *,
    reference_radius: float,
    max_step: float,
) -> dict[str, object]:
    """Report whether the reference and step boxes intersect in every joint.

    The calculated minimum jump is only a geometric lower bound. A motor
    proposal can make a larger jump; satisfying it does not establish balance.
    """
    if (
        not isinstance(previous_target, np.ndarray)
        or not isinstance(reference_target, np.ndarray)
        or previous_target.ndim != 1
        or previous_target.size == 0
        or previous_target.shape != reference_target.shape
        or previous_target.dtype.kind != "f"
        or reference_target.dtype.kind != "f"
        or not np.isfinite(previous_target).all()
        or not np.isfinite(reference_target).all()
        or type(reference_radius) not in (int, float)
        or type(max_step) not in (int, float)
        or not np.isfinite(reference_radius)
        or not np.isfinite(max_step)
        or reference_radius < 0
        or max_step < 0
    ):
        raise ValueError("finite aligned targets and nonnegative bounds required")
    gap = np.abs(previous_target.astype(np.float64) - reference_target.astype(np.float64))
    minimum = np.maximum(0.0, gap - reference_radius)
    # No rounding slack is added: callers choose explicit physical tolerances.
    return {
        "minimum_jump_rad": minimum,
        "maximum_minimum_jump_rad": float(minimum.max()),
        "limiting_joint_index": int(minimum.argmax()),
        "feasible": bool(np.all(minimum <= max_step)),
        "violating_joint_indices": np.flatnonzero(minimum > max_step),
    }
