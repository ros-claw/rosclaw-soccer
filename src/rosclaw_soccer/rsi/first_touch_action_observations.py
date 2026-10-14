"""Observed action/response diagnostics, explicitly NOT a causal authority test."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS


def observe_action_response(trace: dict[str, Any]) -> dict[str, Any]:
    outcome = label_first_touch(trace)
    shapes = {
        "root_pose_xyzw_m": (300, 1, 7),
        "root_velocity_world": (300, 1, 6),
        "ball_position_before_step_m": (300, 1, 3),
        "ball_linear_velocity_before_step_m_s": (300, 1, 3),
        "foot_geometry_position_before_step_m": (300, 1, 4, 3),
        "navigation_command": (300, 1, 3),
        "joint_target_rad": (300, 1, 29),
        "joint_position_rad": (300, 1, 29),
        "motor_delta_rad": (300, 1, 12),
        "torque_nm": (300, 1, 10, 29),
        "actual_actuator_force_nm": (300, 1, 10, 29),
    }
    arrays = {}
    for name, shape in shapes.items():
        value = np.asarray(trace.get(name))
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError("complete finite float64 observed action/response trace required")
        arrays[name] = value[:, 0]
    q = arrays["root_pose_xyzw_m"][:, 3:7]
    if not np.allclose(np.linalg.norm(q, axis=1), 1, atol=0.01, rtol=0):
        raise ValueError("unit measured root quaternions required")
    x, y, z, w = q.T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    velocity = arrays["root_velocity_world"]
    body_forward = np.cos(yaw) * velocity[:, 0] + np.sin(yaw) * velocity[:, 1]
    foot_ball = (
        arrays["ball_position_before_step_m"][:, None, :]
        - arrays["foot_geometry_position_before_step_m"][:, :2]
    )
    foot_distance = np.linalg.norm(foot_ball, axis=2).min(axis=1)
    stop = outcome.first_foot_contact_frame
    stop = 300 if stop is None else stop
    metrics = None
    if stop:
        region = slice(0, stop)
        torque = arrays["torque_nm"][region]
        metrics = dict(
            recorded_navigation_forward_mean_m_s=float(
                np.mean(arrays["navigation_command"][region, 0])
            ),
            measured_body_forward_mean_m_s=float(np.mean(body_forward[region])),
            measured_ball_world_forward_mean_m_s=float(
                np.mean(arrays["ball_linear_velocity_before_step_m_s"][region, 0])
            ),
            minimum_ankle_body_center_to_ball_center_distance_m=float(
                np.min(foot_distance[region])
            ),
            ankle_center_within_0_95m_frames=int(np.count_nonzero(foot_distance[region] <= 0.95)),
            executed_residual_nonzero_frames=int(
                np.count_nonzero(np.any(arrays["motor_delta_rad"][region] != 0, axis=1))
            ),
            instantaneous_target_minus_observed_joint_rmse_rad=float(
                np.sqrt(
                    np.mean(
                        (arrays["joint_target_rad"][region] - arrays["joint_position_rad"][region])
                        ** 2
                    )
                )
            ),
            recorded_torque_at_hard_limit_fraction=float(
                np.mean(
                    np.isclose(np.abs(torque), np.asarray(G1_HARD_TORQUE_LIMITS), atol=1e-8, rtol=0)
                )
            ),
            recorded_command_minus_actuator_force_rmse_nm=float(
                np.sqrt(np.mean((torque - arrays["actual_actuator_force_nm"][region]) ** 2))
            ),
        )
    return dict(
        schema="soccer.rsi.observed_action_response.v1",
        label=outcome.to_dict(),
        pre_first_foot_contact_frames=stop,
        precontact=metrics,
        causal_authority_proven=False,
        action_intervention_performed=False,
        complete_requested_to_executed_chain_available=False,
        missing_measurements=[
            "preclip_navigation_request",
            "unclipped_motor_residual_request",
            "per_substep_unclipped_pd_request",
            "explicit_motion_reference",
        ],
        distance_is_body_center_proxy_not_contact_geometry=True,
        target_error_is_same_frame_not_tracking_latency=True,
        new_physics_executions=0,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
