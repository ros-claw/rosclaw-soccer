"""Localize measured post-touch velocity changes, without causal attribution.

Joint state snapshots precede each tick. Thus qvel[k+1]-qvel[k] spans the
ten substeps holding target[k], not target[k+1]. Recorded ball contact forces
cover six leg bodies only; zero values do not rule out ground/self contact.
Artifact authentication and independent dynamics replay remain caller duties.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS


def observe_postcontact_action_transition(trace: dict[str, Any]) -> dict[str, Any]:
    outcome = label_first_touch(trace)
    shapes = {
        "joint_velocity_rad_s": (300, 1, 29),
        "joint_target_rad": (300, 1, 29),
        "pre_motor_joint_target_rad": (300, 1, 29),
        "motor_delta_rad": (300, 1, 12),
        "torque_nm": (300, 1, 10, 29),
        "actual_actuator_force_nm": (300, 1, 10, 29),
    }
    optional = {
        "foundation_navigation_joint_target_rad": (300, 1, 29),
        "taskspace_joint_increment_rad": (300, 1, 29),
        "pd_requested_torque_nm": (300, 1, 10, 29),
    }
    present = optional.keys() & trace.keys()
    if present and present != optional.keys():
        raise ValueError("complete optional bounded action-chain capture required")
    arrays = {}
    for key, shape in (shapes | (optional if present else {})).items():
        value = np.asarray(trace.get(key))
        if (
            value.shape != shape
            or value.dtype != np.float64
            or not np.isfinite(value).all()
            or np.max(np.abs(value)) > 1e6
        ):
            raise ValueError("complete bounded finite float64 action measurements required")
        arrays[key] = value[:, 0]
    motor = np.zeros((300, 29), dtype=np.float64)
    motor[:, :12] = arrays["motor_delta_rad"]
    pre = arrays["pre_motor_joint_target_rad"]
    target = arrays["joint_target_rad"]
    if not np.array_equal(pre + motor, target):
        raise ValueError("final targets must equal measured pre-motor plus leg residual")
    if present and not np.allclose(
        arrays["foundation_navigation_joint_target_rad"] + arrays["taskspace_joint_increment_rad"],
        pre,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("captured navigation plus taskspace targets differ from pre-motor")
    limits = np.asarray(G1_HARD_TORQUE_LIMITS)
    if np.any(np.abs(arrays["torque_nm"]) > limits + 1e-8):
        raise ValueError("recorded torque command exceeds canonical hard limits")
    first = outcome.first_foot_contact_frame
    velocity = arrays["joint_velocity_rad_s"]

    def window(last: int, requested: int) -> dict[str, Any] | None:
        if first is None or last <= first:
            return None
        differences = np.diff(velocity[first : last + 1], axis=0)
        norms = np.linalg.norm(differences, axis=1)
        offset = int(np.argmax(norms))  # earliest maximum, not best-looking event
        frame = first + offset
        change = differences[offset]
        joint = int(np.argmax(np.abs(change)))
        components = None
        if frame:
            components = {
                "final_target": target[frame] - target[frame - 1],
                "pre_motor_target": pre[frame] - pre[frame - 1],
                "applied_motor_residual": motor[frame] - motor[frame - 1],
            }
            if present:
                for name, key in (
                    ("foundation_navigation_target", "foundation_navigation_joint_target_rad"),
                    ("taskspace_increment", "taskspace_joint_increment_rad"),
                ):
                    components[name] = arrays[key][frame] - arrays[key][frame - 1]
            components = {
                name: dict(vector_rad=value.tolist(), l2_rad=float(np.linalg.norm(value)))
                for name, value in components.items()
            }
        command = arrays["torque_nm"][frame]
        measured = arrays["actual_actuator_force_nm"][frame]
        requested_torque = arrays["pd_requested_torque_nm"][frame] if present else None
        return dict(
            first_frame=first,
            last_frame=last,
            requested_last_frame=requested,
            complete_requested_window=last == requested,
            body_snapshot_span_s=(last - first) * 0.02,
            peak_velocity_change_interval_start_frame=frame,
            peak_velocity_change_interval_end_frame=frame + 1,
            held_target_frame=frame,
            maximum_joint_velocity_step_l2_change_rad_s=float(norms[offset]),
            joint_velocity_change_vector_rad_s=change.tolist(),
            largest_absolute_velocity_change_joint_index=joint,
            largest_absolute_velocity_change_joint_name=G1_DDS_JOINT_NAMES[joint],
            target_step_into_held_interval=components,
            recorded_ball_leg_force_n=np.asarray(trace["force_n"])[frame, 0].tolist(),
            held_interval_maximum_absolute_torque_command_nm=float(np.max(np.abs(command))),
            held_interval_torque_at_hard_limit_fraction=float(
                np.mean(np.isclose(np.abs(command), limits, rtol=0, atol=1e-8))
            ),
            held_interval_command_minus_actuator_force_rmse_nm=float(
                np.sqrt(np.mean((command - measured) ** 2))
            ),
            held_interval_maximum_absolute_requested_pd_torque_nm=(
                None if requested_torque is None else float(np.max(np.abs(requested_torque)))
            ),
            held_interval_requested_pd_above_hard_limit_fraction=(
                None
                if requested_torque is None
                else float(np.mean(np.abs(requested_torque) > limits))
            ),
        )

    return dict(
        schema="soccer.rsi.postcontact_action_transition_observations.v1",
        outcome=outcome.to_dict(),
        contact_to_plus60=None if first is None else window(min(first + 60, 299), first + 60),
        contact_to_episode_end=None if first is None else window(299, 299),
        executed_target_chain_verified=True,
        bounded_foundation_taskspace_capture_available=bool(present),
        target_component_norms_are_not_additive=True,
        zero_ball_leg_force_does_not_rule_out_ground_or_self_contact=True,
        ground_contact_or_support_phase_observed=False,
        velocity_step_difference_is_not_instantaneous_acceleration_or_jerk=True,
        causal_failure_reason_proven=False,
        input_receipts_authenticated_here=False,
        physics_replayed_here=False,
        task_or_reward_changed=False,
        new_physics_executions=0,
        neural_optimizer_updates=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
