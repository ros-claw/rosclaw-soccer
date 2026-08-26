"""Backend-neutral predictive joint-envelope safety projection."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def project_joint_safe_torque_numpy(
    *,
    joint_position: NDArray[np.float64],
    joint_velocity: NDArray[np.float64],
    commanded_torque: NDArray[np.float64],
    joint_ranges: NDArray[np.float64],
    limited: NDArray[np.bool_],
    margin_rad: float = 0.05,
    prediction_horizon_sec: float = 0.08,
    boundary_kp: float = 80.0,
    boundary_kd: float = 6.0,
) -> tuple[NDArray[np.float64], bool]:
    """NumPy equivalent used by the dependency-minimal CPU physics exam."""

    if (
        joint_position.shape != (29,)
        or joint_velocity.shape != (29,)
        or commanded_torque.shape != (29,)
        or joint_ranges.shape != (29, 2)
        or limited.shape != (29,)
    ):
        raise ValueError("NumPy joint guard requires the 29-DoF G1 contract")
    if not all(
        np.all(np.isfinite(value))
        for value in (joint_position, joint_velocity, commanded_torque, joint_ranges)
    ):
        raise ValueError("NumPy joint guard inputs must be finite")
    if not 0.0 < margin_rad <= 0.10 or not 0.0 < prediction_horizon_sec <= 0.20:
        raise ValueError("NumPy joint guard envelope parameters are invalid")
    predicted = joint_position + prediction_horizon_sec * joint_velocity
    lower = joint_ranges[:, 0] + margin_rad
    upper = joint_ranges[:, 1] - margin_rad
    lower_threat = limited & (predicted < lower)
    upper_threat = limited & (predicted > upper)
    lower_brake = boundary_kp * (lower - joint_position) - boundary_kd * joint_velocity
    upper_brake = boundary_kp * (upper - joint_position) - boundary_kd * joint_velocity
    projected = commanded_torque.copy()
    projected[lower_threat] = np.maximum(projected[lower_threat], lower_brake[lower_threat])
    projected[upper_threat] = np.minimum(projected[upper_threat], upper_brake[upper_threat])
    return projected, bool(np.any(lower_threat | upper_threat))


def project_joint_safe_torque_torch(
    *,
    joint_position: Any,
    joint_velocity: Any,
    commanded_torque: Any,
    joint_ranges: Any,
    limited: Any,
    margin_rad: float = 0.05,
    prediction_horizon_sec: float = 0.08,
    boundary_kp: float = 80.0,
    boundary_kd: float = 6.0,
) -> tuple[Any, Any]:
    """Project only outward torque when a predicted joint limit is threatened."""

    import torch

    if joint_position.ndim != 2 or joint_position.shape[1] != 29:
        raise ValueError("torch joint guard requires batched 29-DoF joint positions")
    if (
        joint_velocity.shape != joint_position.shape
        or commanded_torque.shape != joint_position.shape
    ):
        raise ValueError("torch joint guard state and torque shapes must match")
    if tuple(joint_ranges.shape) != (29, 2) or tuple(limited.shape) != (29,):
        raise ValueError("torch joint guard range contract is invalid")
    if not 0.0 < margin_rad <= 0.10 or not 0.0 < prediction_horizon_sec <= 0.20:
        raise ValueError("torch joint guard envelope parameters are invalid")
    predicted = joint_position + prediction_horizon_sec * joint_velocity
    lower = joint_ranges[:, 0] + margin_rad
    upper = joint_ranges[:, 1] - margin_rad
    lower_threat = limited.unsqueeze(0) & (predicted < lower)
    upper_threat = limited.unsqueeze(0) & (predicted > upper)
    lower_brake = boundary_kp * (lower - joint_position) - boundary_kd * joint_velocity
    upper_brake = boundary_kp * (upper - joint_position) - boundary_kd * joint_velocity
    projected = torch.where(
        lower_threat, torch.maximum(commanded_torque, lower_brake), commanded_torque
    )
    projected = torch.where(upper_threat, torch.minimum(projected, upper_brake), projected)
    return projected, lower_threat | upper_threat


__all__ = ["project_joint_safe_torque_numpy", "project_joint_safe_torque_torch"]
