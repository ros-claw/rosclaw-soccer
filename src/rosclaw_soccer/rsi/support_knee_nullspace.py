"""SIM_ONLY support-knee clearance without commanding support-foot translation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

RETRACT_DISTANCES_M = (0.0, 0.04, 0.08)
MAX_JOINT_DELTA_RAD = 0.18
MAX_PREDICTED_FOOT_DRIFT_M = 0.005


@dataclass(frozen=True)
class SupportKneeNullspaceProposal:
    joint_delta_rad: tuple[float, ...]
    predicted_knee_retract_m: float
    predicted_foot_drift_m: float
    abstained: bool


def support_knee_nullspace_delta(
    support_foot_jacobian_m_per_rad: np.ndarray[Any, Any],
    support_knee_x_jacobian_m_per_rad: np.ndarray[Any, Any],
    baseline_joint_target_rad: np.ndarray[Any, Any],
    joint_limits_rad: np.ndarray[Any, Any],
    *,
    retract_m: float,
    support_foot_grounded: bool,
    swing_foot_airborne: bool,
) -> SupportKneeNullspaceProposal:
    """Propose a bounded six-joint change only in the support-foot nullspace.

    The caller must still project joint targets through limits and audit the
    resulting whole-body physics. This helper never grants actuator authority.
    """
    foot = np.asarray(support_foot_jacobian_m_per_rad, dtype=np.float64)
    knee = np.asarray(support_knee_x_jacobian_m_per_rad, dtype=np.float64)
    baseline = np.asarray(baseline_joint_target_rad, dtype=np.float64)
    limits = np.asarray(joint_limits_rad, dtype=np.float64)
    if (
        foot.shape != (3, 6)
        or knee.shape != (6,)
        or baseline.shape != (6,)
        or limits.shape != (6, 2)
        or not np.isfinite(foot).all()
        or not np.isfinite(knee).all()
        or not np.isfinite(baseline).all()
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])
        or type(retract_m) is not float
        or retract_m not in RETRACT_DISTANCES_M
        or type(support_foot_grounded) is not bool
        or type(swing_foot_airborne) is not bool
    ):
        raise ValueError("finite six-joint support geometry and bounded intent required")
    zero = SupportKneeNullspaceProposal((0.0,) * 6, 0.0, 0.0, True)
    if retract_m == 0.0 or not support_foot_grounded or not swing_foot_airborne:
        return zero
    gram = foot @ foot.T + 1e-5 * np.eye(3)
    nullspace_knee = knee - foot.T @ np.linalg.solve(gram, foot @ knee)
    leverage = float(knee @ nullspace_knee)
    if not math.isfinite(leverage) or leverage < 1e-5:
        return zero
    delta = -retract_m * nullspace_knee / (leverage + 1e-4)
    peak = float(np.max(np.abs(delta)))
    if peak > MAX_JOINT_DELTA_RAD:
        delta *= MAX_JOINT_DELTA_RAD / peak
    foot_drift = float(np.linalg.norm(foot @ delta))
    if foot_drift > MAX_PREDICTED_FOOT_DRIFT_M:
        delta *= MAX_PREDICTED_FOOT_DRIFT_M / foot_drift
        foot_drift = float(np.linalg.norm(foot @ delta))
    knee_retract = -float(knee @ delta)
    proposed = baseline + delta
    unsafe_direction = (
        (
            ((baseline >= limits[:, 0]) & (baseline <= limits[:, 1]))
            & ((proposed < limits[:, 0]) | (proposed > limits[:, 1]))
        )
        | ((baseline < limits[:, 0]) & (delta < 0))
        | ((baseline > limits[:, 1]) & (delta > 0))
    )
    if (
        not np.isfinite(delta).all()
        or not math.isfinite(foot_drift)
        or not math.isfinite(knee_retract)
        or knee_retract <= 0
        or foot_drift > MAX_PREDICTED_FOOT_DRIFT_M + 1e-12
        or bool(np.any(unsafe_direction))
    ):
        return zero
    return SupportKneeNullspaceProposal(
        tuple(float(value) for value in delta), knee_retract, foot_drift, False
    )
