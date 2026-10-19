"""Bounded offline impact-action screening values, not a live torque policy.

Caller-owned forecasts advance physics and must charge their complete budget.
This module only proposes actuated-leg offsets and reads predicted quantities.
Initial20ms ball velocity is a diagnostic proxy, never the H300/post60 label.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.physics.g1_joint_task_kinematics import G1JointTaskKinematics
from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecast
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS

CAPS = np.minimum(12.0, 0.2 * np.asarray(G1_HARD_TORQUE_LIMITS))
CAPS.flags.writeable = False
MAX_CANDIDATES = 13


def impact_torque_candidates(
    sample: G1JointTaskKinematics, side: int
) -> list[tuple[str, np.ndarray]]:
    if type(sample) is not G1JointTaskKinematics or type(side) is not int or side not in (0, 1):
        raise ValueError("actual task-Jacobian sample and explicit selected foot required")
    for a in (sample.foot_material_point_jacobian_m_per_rad, sample.foot_rotation_jacobian):
        if (
            not isinstance(a, np.ndarray)
            or a.shape != (2, 3, 29)
            or a.dtype != np.float64
            or not np.isfinite(a).all()
            or np.max(abs(a)) > 1e6
        ):
            raise ValueError("finite canonical actuated-only material-point Jacobians required")
    linear = sample.foot_material_point_jacobian_m_per_rad[side]
    angular = sample.foot_rotation_jacobian[side]
    forward = linear.T @ np.array([40.0, 0.0, 0.0])
    lateral = linear.T @ np.array([0.0, 40.0, 0.0])
    lift = linear.T @ np.array([0.0, 0.0, 40.0])
    yaw = angular.T @ np.array([0.0, 0.0, 4.0])
    wishes = [("baseline", np.zeros(29))]
    for name, vector in (("forward", forward), ("lateral", lateral), ("lift", lift), ("yaw", yaw)):
        wishes.extend(((name + "_plus", vector), (name + "_minus", -vector)))
    wishes.extend(
        (f"lateral{lateral_sign:+d}_yaw{yaw_sign:+d}", lateral_sign * lateral + yaw_sign * yaw)
        for lateral_sign in (-1, 1)
        for yaw_sign in (-1, 1)
    )
    result, seen = [], set()
    for name, wish in wishes:
        # No root, ball, support-leg, waist or upper-body variable is optimized.
        offset = np.zeros(29)
        ids = slice(6 * side, 6 * side + 6)
        offset[ids] = np.clip(wish[ids], -CAPS[ids], CAPS[ids])
        offset[offset == 0] = 0.0
        if offset.tobytes() not in seen:
            result.append((name, offset))
            seen.add(offset.tobytes())
    return result


def impact_prediction_score(forecast: G1PDTorqueForecast, offset_nm: np.ndarray) -> dict[str, Any]:
    if type(forecast) is not G1PDTorqueForecast:
        raise ValueError("owned held-target ten-substep forecast required")
    if (
        not isinstance(offset_nm, np.ndarray)
        or offset_nm.shape != (29,)
        or offset_nm.dtype != np.float64
        or not np.isfinite(offset_nm).all()
        or np.any(abs(offset_nm) > CAPS)
        or type(forecast.nonfoot_robot_contact_over_1n) is not bool
    ):
        raise ValueError("bounded canonical research torque offset required")
    velocity = forecast.canonical_qvel_after_substep
    height = forecast.pelvis_height_solver_epoch_m
    forces = forecast.robot_force_on_ball_world_n
    for a, shape in ((velocity, (10, 41)), (height, (10,)), (forces, (10, 3))):
        if (
            not isinstance(a, np.ndarray)
            or a.shape != shape
            or a.dtype != np.float64
            or not np.isfinite(a).all()
            or np.max(abs(a)) > 1e6
        ):
            raise ValueError("whole finite held-target impact predictions required")
    contact = bool(np.any(np.linalg.norm(forces, axis=1) > 1.0))
    safe = not forecast.nonfoot_robot_contact_over_1n and float(height.min()) >= 0.65
    vx, vy, vz = (float(v) for v in velocity[-1, 35:38])
    # Research objective only: a conservative direction proxy plus sufficient
    # forward speed, with a small dimensionless action preference. No claim of
    # kinetic energy, eventual post60 displacement, naturalness or readiness.
    side_excess = max(abs(vy) - 0.25 * max(vx, 0.0), 0.0)
    forward_shortfall = max(1.0 - vx, 0.0)
    regularizer = 1e-4 * float(np.sum((offset_nm / CAPS) ** 2))
    feasible = bool(safe and contact)
    return dict(
        schema="soccer.rsi.impact_forecast_proxy_score.v1",
        feasible_predicted_impact=feasible,
        rejection_reason="PREDICTED_SAFETY_VETO"
        if not safe
        else None
        if contact
        else "NO_PREDICTED_FOOT_CONTACT",
        proxy_cost=side_excess**2 + forward_shortfall**2 + regularizer if feasible else None,
        predicted_ball_velocity_end_m_s=[vx, vy, vz],
        predicted_minimum_pelvis_z_m=float(height.min()),
        predicted_robot_contact_over1n=contact,
        action_regularizer=regularizer,
        outcome_horizon_sec=0.02,
        final_m0_success_proven=False,
        full_body_stability_after_window_proven=False,
        trained_policy=False,
        runtime_execution_authorized=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
