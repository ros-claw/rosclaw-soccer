"""Fixed10-frame future-reference blend; no state/target/execution authority."""

import numpy as np


def blend_planner_segment(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Quintic positions/joints + shortest-hemisphere normalized quaternion blend.

    First pose equals the old future exactly; frame9 onward equals new exactly.
    Quaternion signs need not be continuous (rotation is sign invariant).
    Input legality is never repaired. This does not constrain the rest of a
    learned segment or certify dynamic stability/contact quality.
    """
    if (
        type(old) is not np.ndarray
        or type(new) is not np.ndarray
        or old.dtype != np.float64
        or new.dtype != np.float64
        or old.ndim != 2
        or old.shape != new.shape
        or old.shape[1] != 36
        or not 10 <= len(old) <= 410
        or not np.isfinite(old).all()
        or not np.isfinite(new).all()
        or np.max(abs(np.linalg.norm(old[:, 3:7], axis=1) - 1)) > 1e-4
        or np.max(abs(np.linalg.norm(new[:, 3:7], axis=1) - 1)) > 1e-4
    ):
        raise ValueError("matching finite float64 future segments with unit quaternions required")
    result = new.copy()
    if not np.array_equal(old, new):
        t = np.arange(10, dtype=np.float64) / 9
        weight = (t**3 * (10 - 15 * t + 6 * t * t))[:, None]
        result[:10] = (1 - weight) * old[:10] + weight * new[:10]
        a, b = old[:10, 3:7], new[:10, 3:7].copy()
        b[np.sum(a * b, axis=1) < 0] *= -1
        quaternion = (1 - weight) * a + weight * b
        norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
        if np.any(norm < 0.5):
            raise ValueError("degenerate shortest-hemisphere blend")
        result[:10, 3:7] = quaternion / norm
        result[0] = old[0]
        result[9:] = new[9:]
    result.setflags(write=False)
    return result
