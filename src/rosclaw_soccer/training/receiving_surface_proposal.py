"""Bounded local surface-distance proposals, never a physical feasibility proof.

The caller supplies independently qualified current-pose distance Jacobians.
This solver changes neither geometry nor execution limits. Its linearized
answer must still be tried through the original simulator and receiving exam.
"""

import numpy as np
from numpy.typing import NDArray


def bounded_surface_correction(
    jacobian: NDArray[np.float64],
    distance_change_m: NDArray[np.float64],
    lower_rad: NDArray[np.float64],
    upper_rad: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve a small damped least-squares problem with projected iterations.

    Bounds are *changes* relative to a caller-owned reference action, not
    absolute motor limits. No floating-base coordinates may be supplied.
    Zero Jacobians yield zero correction; an unattainable target is not success.
    """
    inputs = tuple(
        np.asarray(value) for value in (jacobian, distance_change_m, lower_rad, upper_rad)
    )
    if any(value.dtype.kind not in "fiu" or not np.isfinite(value).all() for value in inputs):
        raise ValueError("finite numeric surface proposal arrays required")
    matrix, target, lower, upper = (value.astype(np.float64, copy=True) for value in inputs)
    if (
        matrix.ndim != 2
        or not 1 <= matrix.shape[0] <= 8
        or not 1 <= matrix.shape[1] <= 29
        or target.shape != (matrix.shape[0],)
        or lower.shape != (matrix.shape[1],)
        or upper.shape != lower.shape
        or np.any(np.abs(matrix) > 2)
        or np.any(np.abs(target) > 0.05)
        or np.any(lower > 0)
        or np.any(upper < 0)
        or np.any(lower < -0.2)
        or np.any(upper > 0.2)
    ):
        raise ValueError(
            "bounded surface gradients, changes and zero-containing joint box required"
        )
    # Positive damping makes the objective strongly convex, including singular
    # and contradictory surface constraints. Fixed iterations are deterministic.
    hessian = matrix.T @ matrix + 0.01**2 * np.eye(matrix.shape[1])
    linear = matrix.T @ target
    step = 1.0 / float(np.linalg.eigvalsh(hessian)[-1])
    delta = np.zeros(matrix.shape[1], dtype=np.float64)
    for _ in range(256):
        delta = np.clip(delta - step * (hessian @ delta - linear), lower, upper)
    if not np.isfinite(delta).all():
        raise ValueError("nonfinite surface correction")
    return delta
