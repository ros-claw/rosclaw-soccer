"""Past-only linear contact opportunity for a bounded simulation teacher.

This is an unlearned geometric proposal. Gait, acceleration and ball impacts
can invalidate constant-velocity extrapolation; it is not a contact guarantee.
"""

import numpy as np


def early_contact_intercept(
    relative_position: np.ndarray, relative_velocity: np.ndarray, *, contact_lead_m: float = 0.0
) -> dict:
    """Estimate closest approach to the observed foot point within 0.6 s."""
    if (
        type(contact_lead_m) is not float
        or not 0 <= contact_lead_m <= 0.12
        or any(
            not isinstance(x, np.ndarray)
            or x.dtype != np.float64
            or x.shape != (3,)
            or not np.isfinite(x).all()
            for x in (relative_position, relative_velocity)
        )
    ):
        raise ValueError("finite measured 3D position and velocity vectors required")
    speed2 = float(relative_velocity @ relative_velocity)
    closing = float(relative_position @ relative_velocity) < 0.0
    raw_eta = -float(relative_position @ relative_velocity) / speed2 if speed2 > 0.04 else 0.0
    eta = float(np.clip(raw_eta, 0.0, 0.6))
    miss_vector = relative_position + eta * relative_velocity
    miss = float(np.linalg.norm(miss_vector))
    displacement = miss_vector.copy()
    if contact_lead_m and speed2 > 0.04:
        # Place the foot toward the incoming side, rather than at the predicted
        # ball center. This is a proposal, not an actual collision prediction.
        displacement -= contact_lead_m * relative_velocity / np.sqrt(speed2)
    return {
        "eligible": bool(closing and speed2 > 0.04 and 0.03 <= raw_eta <= 0.6 and miss <= 0.55),
        "eta_s": eta,
        "miss_m": miss,
        "displacement": displacement,
    }
