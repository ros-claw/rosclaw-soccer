"""Bounded contact-surface IK hypothesis; not a qualified football teacher.

Unlike the body-origin probe, the supplied point and its Jacobian must refer
to the SAME actual convex collision surface and current kinematic state.
Caller owns point measurement, swing/support selection and evidence binding.
This function does not choose a foot, step physics or infer a future outcome.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.taskspace_swing_probe import (
    MAX_JOINT_DELTA_RAD,
    REGULARIZATION,
    validate_continuous_swing_parameters,
)


def surface_strike_joint_delta(
    surface_point_world: np.ndarray[Any, Any],
    ball_center_world: np.ndarray[Any, Any],
    surface_point_jacobian: np.ndarray[Any, Any],
    baseline_target: np.ndarray[Any, Any],
    joint_limits: np.ndarray[Any, Any],
    strike_direction_world: np.ndarray[Any, Any],
    *,
    ball_radius_m: float,
    forward_cap_m: float,
    lateral_cap_m: float,
    vertical_cap_m: float,
) -> np.ndarray[Any, Any]:
    """Aim at the ball surface opposite the intended launch direction.

    Current closest mesh point is held fixed only for this local linear solve.
    A new query must be made each control frame. Reaching this point does not
    guarantee useful impact velocity, balance or shot precision; those require
    the separately declared physical experiment. No strike-through, ball force
    injection or enlarged contact geometry is used here.
    """
    validate_continuous_swing_parameters(
        forward_cap_m=forward_cap_m,
        lateral_cap_m=lateral_cap_m,
        vertical_offset_m=vertical_cap_m,
    )
    arrays = (
        (surface_point_world, (3,)),
        (ball_center_world, (3,)),
        (surface_point_jacobian, (3, 6)),
        (baseline_target, (6,)),
        (joint_limits, (6, 2)),
        (strike_direction_world, (3,)),
    )
    if (
        any(
            raw.shape != shape
            or raw.dtype != np.float64
            or not np.isfinite(raw).all()
            or np.max(np.abs(raw)) > 1e6
            for raw, shape in arrays
        )
        or type(ball_radius_m) is not float
        or not 0.05 <= ball_radius_m <= 0.2
        or vertical_cap_m < 0
        or not np.isclose(np.linalg.norm(strike_direction_world), 1, rtol=0, atol=1e-8)
        or np.any(joint_limits[:, 0] >= joint_limits[:, 1])
    ):
        raise ValueError("finite current surface-point IK with unit strike direction required")
    desired_point = ball_center_world - ball_radius_m * strike_direction_world
    caps = np.asarray((forward_cap_m, lateral_cap_m, vertical_cap_m))
    desired_displacement = np.clip(desired_point - surface_point_world, -caps, caps)
    jac = surface_point_jacobian
    delta = jac.T @ np.linalg.solve(
        jac @ jac.T + REGULARIZATION**2 * np.eye(3), desired_displacement
    )
    delta = np.clip(delta, -MAX_JOINT_DELTA_RAD, MAX_JOINT_DELTA_RAD)
    proposed = baseline_target + delta
    safe = (
        (
            (baseline_target >= joint_limits[:, 0])
            & (baseline_target <= joint_limits[:, 1])
            & (proposed >= joint_limits[:, 0])
            & (proposed <= joint_limits[:, 1])
        )
        | ((baseline_target < joint_limits[:, 0]) & (delta > 0))
        | ((baseline_target > joint_limits[:, 1]) & (delta < 0))
    )
    result: np.ndarray[Any, Any] = np.where(safe, delta, 0.0)
    if not np.isfinite(result).all():
        raise ValueError("finite bounded surface-point joint increment required")
    return result
