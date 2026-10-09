"""Vectorized reference for before-action football history, not control authority.

Deliberately independent of the production state helper: cumulative event arrays
and an explicit one-frame shift replace its mutable per-frame event loop. This
component verifies data semantics only, never physics, learning or promotion.
"""

from typing import Any

import numpy as np


def reconstruct_before_action_state(trace: Any) -> np.ndarray[Any, Any]:
    shapes = {
        "force_n": (300, 1, 6),
        "pelvis_z_per_substep_m": (300, 1, 10),
        "ball_position_after_step_m": (300, 1, 3),
    }
    try:
        arrays = {key: np.asarray(trace[key]) for key in shapes}
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("complete independently bound physical trace required") from error
    if any(
        value.shape != shapes[key] or value.dtype != np.float64 or not np.isfinite(value).all()
        for key, value in arrays.items()
    ) or np.any(arrays["force_n"] < 0):
        raise ValueError("complete finite float64 physical histories required")
    force = arrays["force_n"][:, 0]
    height = arrays["pelvis_z_per_substep_m"][:, 0]
    ball = arrays["ball_position_after_step_m"][:, 0]
    contact_seen = np.maximum.accumulate(np.any(force > 1, axis=1))
    dirty_seen = np.maximum.accumulate(np.any(force[:, 2:] > 1, axis=1))
    outside_seen = np.maximum.accumulate(np.abs(ball[:, 1]) > 4)
    unsafe_seen = np.minimum.accumulate(height.min(axis=1)) < 0.65
    frames = np.arange(300)
    elapsed = np.zeros(300)
    events = np.flatnonzero(np.any(force > 1, axis=1))
    if len(events):
        first = int(events[0])
        later = frames > first
        elapsed[later] = (frames[later] - first) / 300
    result = np.column_stack(
        (
            (300 - frames) / 300,
            np.r_[False, contact_seen[:-1]],
            elapsed,
            np.r_[False, dirty_seen[:-1]],
            np.r_[False, outside_seen[:-1]],
            np.r_[False, unsafe_seen[:-1]],
        )
    )
    result.flags.writeable = False
    return result
