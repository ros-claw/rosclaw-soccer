"""Strengthen predicted outward joint braking without changing physical limits.

This numerical simulation helper has no G1 indices or football coordinates.
The owning simulator must still apply its original hard torque cap afterwards.
It is not a certified stopping-distance guarantee or a hardware executor.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def strengthen_outward_joint_braking(
    *,
    joint_position: NDArray[np.floating],
    joint_velocity: NDArray[np.floating],
    projected_torque: NDArray[np.floating],
    joint_ranges: NDArray[np.floating],
    limited: NDArray[np.bool_],
    selected_joints: tuple[int, ...],
    damping: float,
    margin_rad: float = 0.04,
    prediction_horizon_sec: float = 0.08,
    boundary_kp: float = 80.0,
) -> NDArray[np.float64]:
    """Change only selected limited joints moving toward a predicted boundary.

    Lower-bound threats may only increase their existing inward torque;
    upper-bound threats may only decrease it. Inward-moving joints and all
    unselected joints retain their prior projected torque exactly.
    """
    values = (joint_position, joint_velocity, projected_torque, joint_ranges)
    if any(
        not isinstance(value, np.ndarray)
        or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
        or not np.isfinite(value).all()
        or (np.abs(value) > 1e6).any()
        for value in values
    ):
        raise ValueError("finite numeric joint projection arrays required")
    count = joint_position.size
    if (
        not 1 <= count <= 256
        or any(value.shape != (count,) for value in values[:3])
        or joint_ranges.shape != (count, 2)
        or not isinstance(limited, np.ndarray)
        or limited.dtype != np.bool_
        or limited.shape != (count,)
        or type(selected_joints) is not tuple
        or not selected_joints
        or any(type(index) is not int or not 0 <= index < count for index in selected_joints)
        or len(set(selected_joints)) != len(selected_joints)
        or any(
            type(value) not in (float, int) or not math.isfinite(value)
            for value in (damping, margin_rad, prediction_horizon_sec, boundary_kp)
        )
        or not 6.0 <= damping <= 30.0
        or not 0.0 < margin_rad <= 0.2
        or not 0.0 < prediction_horizon_sec <= 0.2
        or not 0.0 < boundary_kp <= 1000.0
        or np.any(limited & (joint_ranges[:, 1] - joint_ranges[:, 0] <= 2 * margin_rad))
    ):
        raise ValueError("bounded named joint projection configuration required")
    q = joint_position.astype(np.float64)
    dq = joint_velocity.astype(np.float64)
    ranges = joint_ranges.astype(np.float64)
    result = projected_torque.astype(np.float64)
    predicted = q + prediction_horizon_sec * dq
    lower, upper = ranges[:, 0] + margin_rad, ranges[:, 1] - margin_rad
    for index in selected_joints:
        if not limited[index]:
            continue
        if predicted[index] < lower[index] and dq[index] < 0:
            brake = boundary_kp * (lower[index] - q[index]) - damping * dq[index]
            result[index] = max(result[index], brake)
        elif predicted[index] > upper[index] and dq[index] > 0:
            brake = boundary_kp * (upper[index] - q[index]) - damping * dq[index]
            result[index] = min(result[index], brake)
    return result
