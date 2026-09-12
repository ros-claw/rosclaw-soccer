"""Offline first forward-plane crossing with signed lateral error.

This is geometry, not a success gate. Callers still need contact attribution,
speed, safety and task-specific acceptance. No robot or simulator dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DirectedPlaneCrossing:
    crossed: NDArray[np.bool_]
    segment_index: NDArray[np.int64]
    fraction: NDArray[np.float64]
    signed_lateral_error: NDArray[np.float64]


def first_directed_plane_crossing(
    positions: NDArray[np.floating],
    *,
    origin: NDArray[np.floating],
    forward: NDArray[np.floating],
) -> DirectedPlaneCrossing:
    """Interpolate the first negative-to-nonnegative crossing for each trace.

    ``positions`` includes the initial sample and has shape (time, batch, 2).
    A trace starting on/in front of the plane has not crossed unless it later
    returns behind and crosses forward. Positive error is left of ``forward``.
    No crossing is explicit: crossed=False, segment_index=-1, both scalars NaN.
    These sentinels must never be interpreted as a passing task result.
    """
    if (
        not isinstance(positions, np.ndarray)
        or positions.ndim != 3
        or positions.shape[2] != 2
        or positions.shape[0] < 2
        or positions.shape[1] < 1
        or positions.size > 20_000_000
    ):
        raise ValueError("bounded (time>=2, batch>=1, 2) positions required")
    for value in (positions, origin, forward):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not np.isfinite(value).all()
            or (np.abs(value) > 1e6).any()
        ):
            raise ValueError("bounded finite floating-point geometry required")
    if origin.shape != (2,) or forward.shape != (2,):
        raise ValueError("one planar origin and forward vector required")
    norm = float(np.linalg.norm(forward.astype(np.float64)))
    if norm < 1e-12:
        raise ValueError("nonzero forward vector required")
    axis = forward.astype(np.float64) / norm
    lateral = np.array([-axis[1], axis[0]])
    relative = positions.astype(np.float64) - origin.astype(np.float64)
    distance = relative @ axis
    crossings = (distance[:-1] < 0) & (distance[1:] >= 0)
    crossed = np.asarray(crossings.any(axis=0), dtype=np.bool_)
    first = crossings.argmax(axis=0)
    columns = np.arange(positions.shape[1])
    fraction = np.full(positions.shape[1], np.nan)
    signed = np.full(positions.shape[1], np.nan)
    selected = columns[crossed]
    start = first[crossed]
    before = distance[start, selected]
    after = distance[start + 1, selected]
    fraction[crossed] = -before / (after - before)
    points = relative[start, selected] + fraction[crossed, None] * (
        relative[start + 1, selected] - relative[start, selected]
    )
    signed[crossed] = points @ lateral
    segment = np.where(crossed, first, -1).astype(np.int64)
    for output in (crossed, segment, fraction, signed):
        output.flags.writeable = False
    return DirectedPlaneCrossing(crossed, segment, fraction, signed)
