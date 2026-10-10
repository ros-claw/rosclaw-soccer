"""Explicit SIM diagnostic intervention; not learned or deployed control.

Keep the original command law bit-for-bit by default. The alternative changes
only the lateral acquisition window; it does not change speed, weights, ball
dimensions, contact thresholds, torque limits or taskspace swing eligibility.
"""

import math

import numpy as np


def native_lateral_command(
    *, gap_x: float, gap_y: float, tracking: bool, before_contact: bool, mode: str = "legacy"
) -> float:
    if (
        mode not in ("legacy", "continuous_lateral_diagnostic")
        or type(mode) is not str
        or type(tracking) is not bool
        or type(before_contact) is not bool
        or any(type(v) is not float or not math.isfinite(v) for v in (gap_x, gap_y))
    ):
        raise ValueError("explicit finite SIM lateral diagnostic inputs required")
    active = tracking and gap_x > 0.95 and before_contact if mode == "legacy" else before_contact
    return float(np.clip(1.2 * gap_y, -0.2, 0.2)) if active else 0.0
