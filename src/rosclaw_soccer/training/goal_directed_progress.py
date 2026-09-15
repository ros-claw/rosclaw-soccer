"""Offline signed progress toward contemporaneous targets, not activity credit.

Pure geometry, with no robot, football, simulator or policy dependency. The
caller owns sample-clock alignment, target provenance and physical continuity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def goal_directed_progress(
    *,
    positions: NDArray[np.floating],
    transition_targets: NDArray[np.floating],
    valid_transitions: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Measure each physical displacement against one unchanged target.

    ``positions`` contains N+1 samples; the other arrays contain N targets and
    masks applying to transitions i→i+1. Positive is closer, negative is farther.
    Target motion alone cannot create progress: both distances use target[i].
    A closed path around a fixed target sums to zero (up to rounding), unlike
    cumulative path length or summing only positive approach distances.

    Invalid transitions return NaN, never fabricated zero progress. Callers must
    report coverage and must not count unobserved/invalid transitions as success.
    Changing targets can still reward chasing; this is not proof of task success,
    efficient tactics, stable locomotion, no teleportation or policy promotion.
    """
    for value in (positions, transition_targets):
        if (
            not isinstance(value, np.ndarray)
            or value.ndim != 2
            or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or value.size > 2_000_000
            or not np.isfinite(value).all()
            or (np.abs(value) > 1e6).any()
        ):
            raise ValueError("bounded finite floating-point position and target arrays required")
    count, dimensions = positions.shape
    if (
        count < 2
        or not 1 <= dimensions <= 3
        or transition_targets.shape != (count - 1, dimensions)
        or not isinstance(valid_transitions, np.ndarray)
        or valid_transitions.dtype != np.dtype("bool")
        or valid_transitions.shape != (count - 1,)
    ):
        raise ValueError("N+1 positions, N aligned targets and N explicit boolean masks required")
    points = positions.astype(np.float64)
    targets = transition_targets.astype(np.float64)
    result: NDArray[np.float64] = np.linalg.norm(points[:-1] - targets, axis=1) - np.linalg.norm(
        points[1:] - targets, axis=1
    )
    result[~valid_transitions] = np.nan
    result.flags.writeable = False
    return result
