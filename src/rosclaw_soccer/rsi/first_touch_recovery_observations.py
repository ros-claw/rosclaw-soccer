"""Measured post-touch motion proxies, not naturalness or readiness admission.

Uses recorded pre-step body snapshots. No thresholds, reward changes, inferred
ground contacts, fabricated tails, new simulation or control are introduced.
Artifact authentication and independent qualification remain the caller's job.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch


def observe_first_touch_recovery(trace: dict[str, Any]) -> dict[str, Any]:
    outcome = label_first_touch(trace)
    arrays = {}
    for key, shape in (
        ("root_pose_xyzw_m", (300, 1, 7)),
        ("root_velocity_world", (300, 1, 6)),
        ("joint_velocity_rad_s", (300, 1, 29)),
    ):
        raw = np.asarray(trace.get(key))
        if raw.shape != shape or raw.dtype != np.float64 or not np.isfinite(raw).all():
            raise ValueError("complete finite float64 recovery measurements required")
        arrays[key] = raw[:, 0]
    quaternion = arrays["root_pose_xyzw_m"][:, 3:7]
    norm_squared = np.sum(quaternion**2, axis=1)
    if not np.allclose(norm_squared, 1.0, rtol=0, atol=1e-6):
        raise ValueError("unit measured recovery quaternions required")
    tilt = np.arccos(np.clip(1 - 2 * np.sum(quaternion[:, :2] ** 2, axis=1) / norm_squared, -1, 1))
    first = outcome.first_foot_contact_frame

    def window(last: int, requested_last: int) -> dict[str, Any] | None:
        if first is None or last <= first:
            return None
        region = slice(first, last + 1)
        root = arrays["root_pose_xyzw_m"][region]
        velocity = arrays["root_velocity_world"][region]
        joint_velocity = arrays["joint_velocity_rad_s"][region]
        drawdown = np.maximum.accumulate(root[:, 0]) - root[:, 0]
        return dict(
            first_frame=first,
            last_frame=last,
            requested_last_frame=requested_last,
            complete_requested_window=last == requested_last,
            pre_step_body_snapshots=last - first + 1,
            body_snapshot_span_s=(last - first) * 0.02,
            root_world_x_net_displacement_m=float(root[-1, 0] - root[0, 0]),
            root_world_x_maximum_drawdown_m=float(np.max(drawdown)),
            root_world_x_minimum_recorded_velocity_m_s=float(np.min(velocity[:, 0])),
            root_world_y_range_m=float(np.ptp(root[:, 1])),
            root_world_z_range_m=float(np.ptp(root[:, 2])),
            root_maximum_upright_tilt_rad=float(np.max(tilt[region])),
            root_maximum_recorded_world_angular_speed_rad_s=float(
                np.max(np.linalg.norm(velocity[:, 3:6], axis=1))
            ),
            maximum_absolute_recorded_joint_velocity_rad_s=float(np.max(np.abs(joint_velocity))),
            maximum_joint_velocity_step_l2_change_rad_s=float(
                np.max(np.linalg.norm(np.diff(joint_velocity, axis=0), axis=1))
            ),
        )

    short = None if first is None else window(min(first + 60, 299), first + 60)
    tail = None if first is None else window(299, 299)
    return dict(
        schema="soccer.rsi.first_touch_recovery_observations.v1",
        outcome=outcome.to_dict(),
        contact_to_plus60=short,
        contact_to_episode_end=tail,
        body_snapshot_epoch="before_control_step; first frame precedes its contact interval",
        velocity_epoch="recorded source world-velocity refresh protocol; no resampling",
        drawdown_definition="max preceding world-x root position minus later world-x position",
        world_x_retreat_is_not_body_frame_backpedalling=True,
        velocity_step_difference_is_not_instantaneous_acceleration_or_jerk=True,
        naturalness_or_successor_readiness_qualified=False,
        ground_contact_or_step_count_inferred=False,
        input_receipts_authenticated_here=False,
        physics_replayed_here=False,
        causal_failure_reason_proven=False,
        task_or_reward_changed=False,
        new_physics_executions=0,
        neural_optimizer_updates=0,
        promotion_authorized=False,
        hardware_authorized=False,
    )
