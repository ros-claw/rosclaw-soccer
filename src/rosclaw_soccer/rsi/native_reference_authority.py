"""Privileged contact-reference action-authority probe, not a learned policy."""

import numpy as np


def reference_contact_target(
    base: np.ndarray, reference: np.ndarray, perception: np.ndarray, *, gain: float
) -> tuple[np.ndarray, float]:
    for value, shape in ((base, (29,)), (reference, (29,)), (perception, (9,))):
        if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError("finite HG29 targets/reference and measured ball9 required")
    if type(gain) is not float or gain not in (0.0, 0.25, 0.5):
        raise ValueError("fixed predeclared authority arms only")
    distance = float(np.linalg.norm(perception[:2]))
    f = float(np.clip((0.7 - distance) / 0.35, 0, 1))
    gate = f * f * (3 - 2 * f) if np.any(perception) else 0.0
    if gain == 0 or gate == 0:
        return base.copy(), gate
    # Bound correction rather than repairing a joint-limit violation. The
    # physical caller must reject out-of-range final targets without clipping.
    return base + gate * np.clip(gain * (reference - base), -0.25, 0.25), gate
