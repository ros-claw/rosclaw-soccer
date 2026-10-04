"""Reframe recorded low-latency prior commands using current body orientation.

This reconstructs stored reference rotations, not a live planner or future
physical state. Original encoder layout and its six trailing zeros stay intact.
"""

from typing import Any

import numpy as np


def _rotation(q: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    w, x, y, z = q / np.linalg.norm(q)
    result: np.ndarray[Any, Any] = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    return result


def reframe_low_latency_prior(
    features: Any, source_root_wxyz: Any, current_root_wxyz: Any
) -> np.ndarray[Any, Any]:
    """Owned one-lane 640-vector; frozen joints, current relative rotations.

    The source stores two rotation columns for nine references; the third is
    their cross product. This geometric reconstruction has float32 roundoff,
    not universal encoder/quantizer parity. Identical roots preserve all bits.
    Only low_latency, non-heading-normalized legacy layout is supported.
    """
    values = np.asarray(features)
    old, new = [np.asarray(q, dtype=np.float64) for q in (source_root_wxyz, current_root_wxyz)]
    if (
        values.shape != (640,)
        or values.dtype != np.float32
        or old.shape != (4,)
        or new.shape != (4,)
        or not all(np.isfinite(q).all() for q in (values, old, new))
        or any(abs(float(np.linalg.norm(q)) - 1) > 1e-4 for q in (old, new))
        or np.any(values[634:] != 0)
    ):
        raise ValueError("complete finite low-latency legacy prior and unit roots required")
    stored = values[580:634].astype(np.float64).reshape(9, 6)
    columns = np.stack((stored[:, [0, 2, 4]], stored[:, [1, 3, 5]]), axis=2)
    if (
        np.max(np.abs(np.linalg.norm(columns, axis=1) - 1)) > 1e-3
        or np.max(np.abs(np.sum(columns[:, :, 0] * columns[:, :, 1], axis=1))) > 1e-3
    ):
        raise ValueError("stored reference rotation columns are not orthonormal")
    result: np.ndarray[Any, Any] = values.copy()
    if not np.array_equal(old, new) and not np.array_equal(old, -new):
        third = np.cross(columns[:, :, 0], columns[:, :, 1])[:, :, None]
        relative = np.concatenate((columns, third), axis=2)
        updated = np.einsum("ij,njk->nik", _rotation(new).T @ _rotation(old), relative)
        result[580:634] = (
            updated[:, [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]].reshape(54).astype(np.float32)
        )
    result.flags.writeable = False
    return result
