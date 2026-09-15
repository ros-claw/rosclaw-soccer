"""Measured-motion preview planes for peers of an executing motor option.

A short bounded constant-velocity forecast is a navigation aid, not a physical
collision guarantee. It adds constraints without weakening current-position
clearance. No role, ball, goal, model state or actuator authority lives here.
"""

from __future__ import annotations

import math

import numpy as np


def motor_clearance_planes(
    offsets: np.ndarray, velocities: np.ndarray, *, horizon_sec: float
) -> np.ndarray:
    if (
        not isinstance(offsets, np.ndarray)
        or not isinstance(velocities, np.ndarray)
        or offsets.ndim != 2
        or offsets.shape[1] != 2
        or velocities.shape != offsets.shape
        or len(offsets) > 8
        or offsets.dtype.kind != "f"
        or velocities.dtype.kind != "f"
        or not np.isfinite(offsets).all()
        or not np.isfinite(velocities).all()
        or type(horizon_sec) not in (float, int)
        or not math.isfinite(horizon_sec)
        or not 0.1 <= horizon_sec <= 0.5
    ):
        raise ValueError("finite bounded motor-peer motion preview required")
    planes = []
    for offset, velocity in zip(offsets, velocities, strict=True):
        speed = float(np.linalg.norm(velocity))
        velocity = velocity * min(1.0, 2.0 / max(speed, 1e-9))
        predicted = offset + horizon_sec * velocity
        distance = float(np.linalg.norm(predicted))
        current_distance = float(np.linalg.norm(offset))
        # Do not constrain a stationary or receding peer more than the existing
        # current-position shield. Preserve exact legacy commands in that case.
        if distance >= min(1.2, current_distance) or current_distance < 1e-9:
            continue
        normal = -predicted / distance if distance > 1e-9 else -offset / current_distance
        planes.append((float(normal[0]), float(normal[1]), 0.6 * (0.8 - distance)))
    return np.asarray(planes, dtype=np.float64).reshape((-1, 3))
