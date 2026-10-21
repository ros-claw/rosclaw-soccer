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
    # Bound correction, never repair/clip the foundation's original PD target.
    # Caller rejects any introduced or worsened target-range violation.
    return base + gate * np.clip(gain * (reference - base), -0.25, 0.25), gate


def worsens_target_range(base: np.ndarray, target: np.ndarray, bounds: np.ndarray) -> bool:
    """Original PD targets may exceed position limits; allow no *new* excess."""
    if (
        base.shape != (29,)
        or target.shape != (29,)
        or bounds.shape != (29, 2)
        or not all(np.isfinite(x).all() for x in (base, target, bounds))
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError("finite HG29 target and ordered range required")

    def excess(value):
        return np.maximum(bounds[:, 0] - value, 0) + np.maximum(value - bounds[:, 1], 0)

    return bool(np.any(excess(target) > excess(base)))
