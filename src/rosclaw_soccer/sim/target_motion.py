"""Measured planar motion toward a declared target; not accuracy or action authority.

Training and replay must use the same target frame. A field-axis speed is not
a substitute for this projection. Velocity increments use the *same current*
direction for both velocities, so turning the target ray earns no impulse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TargetMotion:
    distance_m: NDArray[np.float64]
    directed_speed_mps: NDArray[np.float64]
    directed_increment_mps: NDArray[np.float64]
    direction_resolved: NDArray[np.bool_]


def measure_target_motion(
    position_xy: NDArray[np.floating],
    velocity_xy: NDArray[np.floating],
    previous_velocity_xy: NDArray[np.floating],
    target_xy: NDArray[np.floating],
) -> TargetMotion:
    """Project bounded measured batches, with no broadcasting or input mutation.

    All inputs have shape (N, 2), N in 1..4096, in one common frame. Inside
    1 cm the ray is attenuated rather than divided by a near-zero distance;
    ``direction_resolved`` is false there. This matches existing strike audits.
    No contact attribution, goal crossing, safety, or success is inferred here.
    """
    arrays = (position_xy, velocity_xy, previous_velocity_xy, target_xy)
    shape = position_xy.shape if isinstance(position_xy, np.ndarray) else None
    if (
        shape is None
        or len(shape) != 2
        or not 1 <= shape[0] <= 4096
        or shape[1] != 2
        or any(
            not isinstance(a, np.ndarray)
            or a.shape != shape
            or a.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not np.isfinite(a).all()
            or np.any(np.abs(a) > 1e6)
            for a in arrays
        )
    ):
        raise ValueError("finite bounded equal (N, 2) float32/64 arrays required")
    position, velocity, previous, target = (a.astype(np.float64, copy=True) for a in arrays)
    ray = target - position
    distance = np.linalg.norm(ray, axis=1)
    direction = ray / np.maximum(distance[:, None], 0.01)
    speed = np.sum(velocity * direction, axis=1)
    increment = np.sum((velocity - previous) * direction, axis=1)
    resolved = distance >= 0.01
    for result in (distance, speed, increment, resolved):
        result.flags.writeable = False
    return TargetMotion(distance, speed, increment, resolved)
