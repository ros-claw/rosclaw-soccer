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


def observe_contact_transition(trace: dict[str, Any]) -> dict[str, Any]:
    """Measured transition timing, not a causal intervention or physics replay.

    Commands are the recorded world-frame navigation reference. Their delta
    across two control ticks is not a measured root acceleration.
    """
    outcome = label_first_touch(trace)
    navigation = np.asarray(trace.get("navigation_command"))
    if (
        navigation.shape != (300, 1, 3)
        or navigation.dtype != np.float64
        or not np.isfinite(navigation).all()
    ):
        raise ValueError("complete finite float64 navigation measurements required")
    forces = np.asarray(trace["force_n"])[:, 0]
    heights = np.asarray(trace["pelvis_z_per_substep_m"])[:, 0]
    any_contacts = np.flatnonzero(np.any(forces > 1.0, axis=1))
    invalid = np.flatnonzero(np.any(forces[:, 2:] > 1.0, axis=1))
    low = np.flatnonzero(np.any(heights < 0.65, axis=1))
    first = int(any_contacts[0]) if len(any_contacts) else None
    first_invalid = int(invalid[0]) if len(invalid) else None
    body_names = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_ankle_pitch_link",
        "right_ankle_pitch_link",
        "left_knee_link",
        "right_knee_link",
    )
    invalid_indices = (
        []
        if first_invalid is None
        else (np.flatnonzero(forces[first_invalid, 2:] > 1.0) + 2).tolist()
    )
    transition = None
    if first is not None and first < 299:
        transition = (navigation[first + 1, 0] - navigation[first, 0]).tolist()
    return dict(
        schema="soccer.rsi.observed_contact_transition.v1",
        outcome=outcome.to_dict(),
        first_any_contact_frame=first,
        first_nonfoot_contact_frame=first_invalid,
        first_nonfoot_contact_bodies=[body_names[i] for i in invalid_indices],
        first_low_pelvis_frame=int(low[0]) if len(low) else None,
        foot_preceded_nonfoot=(
            None
            if outcome.first_foot_contact_frame is None or first_invalid is None
            else outcome.first_foot_contact_frame < first_invalid
        ),
        frames_foot_to_nonfoot=(
            None
            if outcome.first_foot_contact_frame is None or first_invalid is None
            else first_invalid - outcome.first_foot_contact_frame
        ),
        maximum_nonfoot_contact_force_n=float(np.max(forces[:, 2:])),
        contact_transition_navigation_delta_m_s=transition,
        control_dt_s=0.02,
        command_delta_is_not_measured_root_acceleration=True,
        new_physics_executions=0,
        causal_intervention_performed=False,
        causation_proven=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )


def observe_postcontact_separation(trace: dict[str, Any]) -> dict[str, Any]:
    """World-frame, pre-step snapshots around measured contact events.

    Frame f+1 is the first available pre-step snapshot after contact in f.
    Body-center distances are descriptive proxies, not collision clearances.
    A horizon boundary remains missing rather than copied or extrapolated.
    """
    transition = observe_contact_transition(trace)
    arrays = {}
    for name, shape in (
        ("root_pose_xyzw_m", (300, 1, 7)),
        ("root_velocity_world", (300, 1, 6)),
        ("ball_position_before_step_m", (300, 1, 3)),
        ("ball_linear_velocity_before_step_m_s", (300, 1, 3)),
    ):
        value = np.asarray(trace.get(name))
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError("complete finite float64 separation measurements required")
        arrays[name] = value[:, 0]

    def snapshot(frame: int | None) -> dict[str, Any] | None:
        if frame is None or not 0 <= frame < 300:
            return None
        gap = arrays["ball_position_before_step_m"][frame] - arrays["root_pose_xyzw_m"][frame, :3]
        relative = (
            arrays["ball_linear_velocity_before_step_m_s"][frame]
            - arrays["root_velocity_world"][frame, :3]
        )
        return dict(
            frame=frame,
            ball_minus_pelvis_world_position_m=gap.tolist(),
            ball_minus_root_world_linear_velocity_m_s=relative.tolist(),
            ball_world_height_m=float(arrays["ball_position_before_step_m"][frame, 2]),
        )

    first = transition["outcome"]["first_foot_contact_frame"]
    nonfoot = transition["first_nonfoot_contact_frame"]
    post = None if first is None else first + 1
    # Do not call a pre-foot nonfoot event a follow-through observation.
    delayed = first is not None and nonfoot is not None and nonfoot > first
    return dict(
        schema="soccer.rsi.observed_postcontact_separation.v1",
        transition=transition,
        pre_foot_contact=snapshot(first),
        first_post_foot_contact=snapshot(post),
        ten_frames_after_foot_contact=snapshot(None if first is None else first + 10),
        pre_delayed_nonfoot_contact=snapshot(nonfoot if delayed else None),
        body_center_proxy_not_collision_clearance=True,
        velocities_are_recorded_world_linear_not_navigation_commands=True,
        velocity_refresh_and_root_reference_inherited_from_source_protocol=True,
        relative_velocity_is_not_finite_difference_of_position_gap=True,
        snapshots_are_before_step=True,
        new_physics_executions=0,
        causation_proven=False,
        training_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
