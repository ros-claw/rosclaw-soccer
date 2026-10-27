"""Causal, body-relative SIM context for a later intervention selector.

No outcome, referee, future state or course identity is an input. This module
does not select or apply a robot action.
"""

import numpy as np

from rosclaw_soccer.rsi.paid_balance_recovery import body_feedback
from rosclaw_soccer.rsi.relay_fall_samples import _rotation


def relay_balance_context(q: np.ndarray, v: np.ndarray, return_goal: np.ndarray) -> np.ndarray:
    if (
        q.shape != (43,)
        or v.shape != (41,)
        or return_goal.shape != (3,)
        or not all(np.isfinite(a).all() for a in (q, v, return_goal))
    ):
        raise ValueError("actual finite canonical body, ball and assigned goal required")
    rotation = _rotation(q[3:7])
    feature = np.concatenate(
        (
            body_feedback(q, v),
            rotation.T @ (q[36:39] - q[:3]),
            rotation.T @ (v[35:38] - v[:3]),
            rotation.T @ (return_goal - q[:3]),
        )
    )
    if feature.shape != (17,) or not np.isfinite(feature).all():
        raise ValueError("finite causal body-relative context required")
    return np.asarray(feature, dtype=np.float64)
