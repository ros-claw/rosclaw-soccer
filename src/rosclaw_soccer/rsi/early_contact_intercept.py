"""Past-only linear contact opportunity for a bounded simulation teacher.

This is an unlearned geometric proposal. Gait, acceleration and ball impacts
can invalidate constant-velocity extrapolation; it is not a contact guarantee.
"""

import numpy as np


def early_contact_intercept(relative_position: np.ndarray, relative_velocity: np.ndarray) -> dict:
    """Estimate closest approach to the observed foot point within 0.6 s."""
    if any(
        not isinstance(x, np.ndarray)
        or x.dtype != np.float64
        or x.shape != (3,)
        or not np.isfinite(x).all()
        for x in (relative_position, relative_velocity)
    ):
        raise ValueError("finite measured 3D position and velocity vectors required")
    speed2 = float(relative_velocity @ relative_velocity)
    closing = float(relative_position @ relative_velocity) < 0.0
    raw_eta = -float(relative_position @ relative_velocity) / speed2 if speed2 > 0.04 else 0.0
    eta = float(np.clip(raw_eta, 0.0, 0.6))
    miss_vector = relative_position + eta * relative_velocity
    miss = float(np.linalg.norm(miss_vector))
    return {
        "eligible": bool(closing and speed2 > 0.04 and 0.03 <= raw_eta <= 0.6 and miss <= 0.55),
        "eta_s": eta,
        "miss_m": miss,
        "displacement": miss_vector,
    }
