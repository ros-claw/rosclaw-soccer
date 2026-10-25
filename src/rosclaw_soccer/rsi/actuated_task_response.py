"""Bounded task-derived HG29 directions using the existing SIM target codec.

This is not a fitted PCA basis, a neural update, a physical result, or extra
actuator authority. Only the first qualified temporal lobe is used.
"""

import numpy as np

from rosclaw_soccer.rsi.executed_target_synergy import ExecutedTargetSynergy


def actuated_task_plan(start_control: int, delta: np.ndarray) -> ExecutedTargetSynergy:
    if (
        type(start_control) is not int
        or not 0 <= start_control <= 239
        or type(delta) is not np.ndarray
        or delta.shape != (29,)
        or delta.dtype != np.float64
        or not np.isfinite(delta).all()
        or np.max(abs(delta)) > 0.03000001
        or np.sum(abs(delta)) > 0.06000001
    ):
        raise ValueError("current H300 boundary and bounded L1 actual-target direction required")
    axis = int(np.argmax(abs(delta)))
    signed_peak = float(delta[axis])
    if signed_peak:
        basis = delta / signed_peak  # Canonical sign: largest basis entry is +1.
    else:
        basis = np.zeros(29, np.float64)
        basis[0] = 1
    plan = ExecutedTargetSynergy(
        start_control,
        (tuple(float(v) for v in basis),),
        ((signed_peak,), (0.0,)),
        projection_mode="per_axis",
    )
    plan.offsets()
    return plan
