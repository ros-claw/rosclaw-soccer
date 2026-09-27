"""Bounded simulation joint-target projection for learned motor residuals."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class JointTargetProjection:
    target_rad: NDArray[np.float64]
    projection_count: int
    maximum_projection_rad: float


def project_modified_joint_targets(
    *,
    target_rad: NDArray[np.float64],
    limits_rad: NDArray[np.float64],
    modified_indices: Sequence[int],
    maximum_allowed_projection_rad: float = 0.25,
) -> JointTargetProjection:
    """Project only learned-action joints; never silently alter frozen joints.

    A source policy may have a nominal target slightly outside the simulator's
    physical joint limits. This projector does not grant more authority to the
    residual: it clamps only explicitly modified joints and rejects projections
    too large to call a bounded correction. The caller must audit the count.
    """
    target = np.asarray(target_rad)
    limits = np.asarray(limits_rad)
    if (
        target.ndim != 1
        or target.dtype.kind not in "fiu"
        or limits.shape != (len(target), 2)
        or limits.dtype.kind not in "fiu"
        or not np.isfinite(target).all()
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])
        or not math.isfinite(maximum_allowed_projection_rad)
        or not 0.0 < maximum_allowed_projection_rad <= 0.25
    ):
        raise ValueError("finite joint targets, ordered limits, and bounded projection required")
    if len(set(modified_indices)) != len(modified_indices) or any(
        type(index) is not int or not 0 <= index < len(target) for index in modified_indices
    ):
        raise ValueError("unique in-range modified joint indices required")
    projected = target.astype(np.float64).copy()
    count = 0
    largest = 0.0
    for index in modified_indices:
        bounded = float(np.clip(projected[index], limits[index, 0], limits[index, 1]))
        difference = abs(float(projected[index]) - bounded)
        if difference > maximum_allowed_projection_rad:
            raise ValueError("learned residual requires excessive joint projection")
        if difference > 1e-9:
            count += 1
            largest = max(largest, difference)
        projected[index] = bounded
    return JointTargetProjection(projected, count, largest)
