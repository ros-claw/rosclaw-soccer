"""Conservative offline projection of reference corrections, not a controller."""

import numpy as np


def project_rate(offsets: np.ndarray, *, max_step: float) -> tuple[np.ndarray, float]:
    """Uniformly shrink finite offsets until every sampled correction step is bounded.

    This preserves zero endpoints and convex joint-position bounds when the
    original source lies inside those bounds. It does not certify actual joint
    velocity, collision geometry, stability, or physical execution.
    """
    if (
        type(offsets) is not np.ndarray
        or offsets.ndim != 2
        or len(offsets) < 2
        or offsets.shape[1] != 6
        or offsets.dtype != np.float64
        or not np.isfinite(offsets).all()
        or type(max_step) is not float
        or not np.isfinite(max_step)
        or max_step <= 0
        or np.any(offsets[[0, -1]] != 0)
    ):
        raise ValueError("finite 6-joint offsets with zero endpoints and positive bound required")
    peak = float(np.max(np.abs(np.diff(offsets, axis=0))))
    scale = min(1.0, max_step / peak) if peak > 0 else 1.0
    projected = offsets * scale
    # Floating rounding must not silently exceed the advertised hard bound.
    if np.max(np.abs(np.diff(projected, axis=0))) > max_step:
        scale = float(np.nextafter(scale, 0.0))
        projected = offsets * scale
    if np.max(np.abs(np.diff(projected, axis=0))) > max_step:
        raise ArithmeticError("projection failed to enforce declared rate bound")
    return projected, scale
