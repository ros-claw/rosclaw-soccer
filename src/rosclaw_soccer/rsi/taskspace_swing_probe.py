"""Bounded SIM_ONLY single-swing-foot task-space first-touch probe."""

from __future__ import annotations

from typing import Any

import numpy as np

FORWARD_CAPS_M = (0.08, 0.16)
MAX_JOINT_DELTA_RAD = 0.35
REGULARIZATION = 0.05
CONTACT_RELEASE_FRAMES = 20


def choose_swing_side(
    feet_xyz: np.ndarray[Any, Any],
    ball_xyz: np.ndarray[Any, Any],
    previous_side: int,
) -> int:
    """Latch an airborne swing foot; never infer one from the future collision."""
    if (
        feet_xyz.shape != (2, 3)
        or ball_xyz.shape != (3,)
        or previous_side not in (-1, 0, 1)
        or not np.isfinite(feet_xyz).all()
        or not np.isfinite(ball_xyz).all()
    ):
        raise ValueError("invalid live foot/ball observation")
    if previous_side >= 0:
        return previous_side
    side = int(np.argmax(feet_xyz[:, 2]))
    if feet_xyz[side, 2] - feet_xyz[1 - side, 2] < 0.02:
        return -1
    gap = float(ball_xyz[0] - feet_xyz[side, 0])
    return side if 0.18 <= gap <= 0.95 else -1


def swing_joint_delta(
    foot_xyz: np.ndarray[Any, Any],
    ball_xyz: np.ndarray[Any, Any],
    linear_jacobian: np.ndarray[Any, Any],
    baseline_target: np.ndarray[Any, Any],
    joint_limits: np.ndarray[Any, Any],
    *,
    forward_cap_m: float,
) -> np.ndarray[Any, Any]:
    """Damped differential IK of one foot, projected into physical joint limits."""
    if (
        foot_xyz.shape != (3,)
        or ball_xyz.shape != (3,)
        or linear_jacobian.shape != (3, 6)
        or baseline_target.shape != (6,)
        or joint_limits.shape != (6, 2)
        or forward_cap_m not in FORWARD_CAPS_M
        or not all(
            np.isfinite(array).all()
            for array in (foot_xyz, ball_xyz, linear_jacobian, baseline_target, joint_limits)
        )
        or np.any(joint_limits[:, 0] >= joint_limits[:, 1])
    ):
        raise ValueError("invalid bounded task-space swing inputs")
    gap = float(ball_xyz[0] - foot_xyz[0])
    if not 0.18 <= gap <= 0.95:
        return np.zeros(6, dtype=np.float64)
    desired = np.array(
        [
            min(max(gap - 0.14, 0.0), forward_cap_m),
            np.clip(ball_xyz[1] - foot_xyz[1], -0.05, 0.05),
            0.0,
        ],
        dtype=np.float64,
    )
    gram = linear_jacobian @ linear_jacobian.T + REGULARIZATION**2 * np.eye(3)
    delta = linear_jacobian.T @ np.linalg.solve(gram, desired)
    ramp = min(1.0, (0.95 - gap) / 0.2)
    delta = np.clip(delta, -MAX_JOINT_DELTA_RAD, MAX_JOINT_DELTA_RAD) * ramp
    proposed = baseline_target + delta
    safe = (
        (
            ((baseline_target >= joint_limits[:, 0]) & (baseline_target <= joint_limits[:, 1]))
            & (proposed >= joint_limits[:, 0])
            & (proposed <= joint_limits[:, 1])
        )
        | ((baseline_target < joint_limits[:, 0]) & (delta > 0))
        | ((baseline_target > joint_limits[:, 1]) & (delta < 0))
    )
    return np.where(safe, delta, 0.0)


def release_joint_delta(
    contact_delta: np.ndarray[Any, Any], frame_since_contact: int
) -> np.ndarray[Any, Any]:
    if (
        contact_delta.shape != (6,)
        or not np.isfinite(contact_delta).all()
        or np.max(np.abs(contact_delta)) > MAX_JOINT_DELTA_RAD + 1e-8
        or type(frame_since_contact) is not int
        or frame_since_contact < 1
    ):
        raise ValueError("invalid causal swing release")
    return contact_delta * max(0.0, 1.0 - frame_since_contact / CONTACT_RELEASE_FRAMES)
