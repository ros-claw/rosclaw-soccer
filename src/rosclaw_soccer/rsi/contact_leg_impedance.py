"""One fixed SIM intervention: half the selected contact leg's stiffness.

Not a learned policy or an authorized controller. Support-leg/trunk stiffness
and all damping remain the caller's original values.
"""

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.taskspace_swing_evidence import LEG_NAMES


def half_contact_leg_stiffness(kp: np.ndarray, side: int | None) -> np.ndarray:
    if (
        kp.shape != (29,)
        or kp.dtype != np.float64
        or not np.isfinite(kp).all()
        or np.any(kp <= 0)
        or np.any(kp > 2000)
        or (side is not None and (type(side) is not int or side not in (0, 1)))
    ):
        raise ValueError("finite original G1 stiffness and explicit causal selected leg required")
    result = kp.copy()
    if side is not None:
        indices = [G1_DDS_JOINT_NAMES.index(name) for name in LEG_NAMES[side]]
        if len(indices) != 6:
            raise ValueError("exact six canonical leg joints required")
        result[indices] *= 0.5
    return result
