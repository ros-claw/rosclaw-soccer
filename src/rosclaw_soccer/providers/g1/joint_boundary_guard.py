"""Velocity-aware SIM-only joint boundary projection for Soccer G1 policies."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from rosclaw.feedback.contracts import canonical_hash

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES


@dataclass(frozen=True)
class G1JointBoundaryGuardConfig:
    protected_joint_names: tuple[str, ...] = (
        "left_knee_joint",
        "left_ankle_roll_joint",
        "waist_pitch_joint",
    )
    margin_rad: float = 0.025
    prediction_horizon_sec: float = 0.05
    boundary_kp: float = 60.0
    boundary_kd: float = 5.0
    minimum_policy_phase: float = 0.0
    maximum_correction_nm: float = 40.0
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.simforge.g1_joint_boundary_guard_config.v1"

    def __post_init__(self) -> None:
        names = tuple(self.protected_joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("joint-boundary guard joints must be non-empty and unique")
        if any(name not in G1_DDS_JOINT_NAMES for name in names):
            raise ValueError("joint-boundary guard references an unknown G1 joint")
        values = (
            self.margin_rad,
            self.prediction_horizon_sec,
            self.boundary_kp,
            self.boundary_kd,
            self.minimum_policy_phase,
            self.maximum_correction_nm,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("joint-boundary guard config must be finite")
        if not 0.005 <= self.margin_rad <= 0.10:
            raise ValueError("joint-boundary guard margin must be in [0.005, 0.10] rad")
        if not 0.01 <= self.prediction_horizon_sec <= 0.20:
            raise ValueError("joint-boundary guard horizon must be in [0.01, 0.20] sec")
        if not 20.0 <= self.boundary_kp <= 200.0:
            raise ValueError("joint-boundary guard kp must be in [20, 200]")
        if not 1.0 <= self.boundary_kd <= 20.0:
            raise ValueError("joint-boundary guard kd must be in [1, 20]")
        if not 0.0 <= self.minimum_policy_phase <= 0.95:
            raise ValueError("joint-boundary guard phase must be in [0, 0.95]")
        if not 1.0 <= self.maximum_correction_nm <= 80.0:
            raise ValueError("joint-boundary guard correction must be in [1, 80] Nm")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("joint-boundary guard is restricted to SIM_ONLY")
        object.__setattr__(self, "protected_joint_names", names)

    @property
    def config_hash(self) -> str:
        return str(canonical_hash(asdict(self)))


def project_g1_joint_boundary_torque(
    *,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    commanded_torque: np.ndarray,
    joint_lower_limits: np.ndarray,
    joint_upper_limits: np.ndarray,
    protected_joint_indices: np.ndarray,
    config: G1JointBoundaryGuardConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = (
        np.asarray(joint_position, dtype=np.float64),
        np.asarray(joint_velocity, dtype=np.float64),
        np.asarray(commanded_torque, dtype=np.float64),
        np.asarray(joint_lower_limits, dtype=np.float64),
        np.asarray(joint_upper_limits, dtype=np.float64),
    )
    indices = np.asarray(protected_joint_indices)
    if any(value.shape != (29,) or not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("joint-boundary projection requires finite 29-DoF vectors")
    if indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in {"i", "u"}:
        raise ValueError("joint-boundary projection requires integer protected indices")
    if np.any(indices < 0) or np.any(indices >= 29) or len(np.unique(indices)) != len(indices):
        raise ValueError("joint-boundary protected indices are invalid")
    position, velocity, torque, lower_limit, upper_limit = arrays
    lower = lower_limit + config.margin_rad
    upper = upper_limit - config.margin_rad
    if np.any(lower[indices] >= upper[indices]):
        raise ValueError("joint-boundary margin collapses a protected joint range")
    predicted = position + config.prediction_horizon_sec * velocity
    protected = np.zeros(29, dtype=np.bool_)
    protected[indices] = True
    lower_threat = protected & (predicted < lower)
    upper_threat = protected & (predicted > upper)
    lower_brake = config.boundary_kp * (lower - position) - config.boundary_kd * velocity
    upper_brake = config.boundary_kp * (upper - position) - config.boundary_kd * velocity
    projected = torque.copy()
    projected[lower_threat] = np.maximum(projected[lower_threat], lower_brake[lower_threat])
    projected[upper_threat] = np.minimum(projected[upper_threat], upper_brake[upper_threat])
    active = lower_threat | upper_threat
    excess = np.maximum(lower - predicted, predicted - upper)
    excess[~active] = 0.0
    return projected, active, excess


__all__ = ["G1JointBoundaryGuardConfig", "project_g1_joint_boundary_torque"]
