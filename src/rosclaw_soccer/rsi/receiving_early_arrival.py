"""Causal SIM research trigger for preparing a receiver, never a catch label."""

import numpy as np


def approaching_contact_window(q: np.ndarray, v: np.ndarray) -> bool:
    """Use current own/ball state only; a true result does not prove possession.

    Canonical state is the 36-position/35-velocity G1 followed by the free ball.
    Fixed research bounds are metres, metres/second and seconds. The estimate
    assumes instantaneous linear relative velocity and is not a trajectory
    forecast or safety check.
    """
    if (
        not isinstance(q, np.ndarray)
        or not isinstance(v, np.ndarray)
        or q.shape != (43,)
        or v.shape != (41,)
        or not np.isfinite(q).all()
        or not np.isfinite(v).all()
    ):
        raise ValueError("actual finite canonical receiver and ball state required")
    delta = q[36:38] - q[:2]
    distance = float(np.linalg.norm(delta))
    if not 0.55 <= distance <= 1.2:
        return False
    relative = v[35:37] - v[:2]
    closing = -float(delta @ relative) / distance
    if closing <= 0.4:
        return False
    return bool(distance / closing <= 0.45)
