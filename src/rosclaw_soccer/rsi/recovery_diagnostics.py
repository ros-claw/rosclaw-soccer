"""Read-only H300 balance/motion diagnostics, never changed task labels.

Six-channel ball-contact veto is NOT proof that the robot fell. Root motion
and angular velocity below use cached20ms body-COM observations; pelvis height
and torque captures retain ten native substeps. No support-margin, naturalness
score, fall detector or physical qualification is inferred from these arrays.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.foot_heading_parity import original_arrays
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_json


def recovery_diagnostics(trace: dict[str, Any]) -> dict[str, Any]:
    original_arrays(trace)
    outcome = label_first_touch(trace).to_dict()
    pose = trace["root_pose_xyzw_m"][:, 0]
    norm = np.linalg.norm(pose[:, 3:], axis=1)
    if np.any(np.abs(norm - 1.0) > 1e-6):
        raise ValueError("finite normalized recorded root quaternions required")
    q = pose[:, 3:] / norm[:, None]
    tilt = np.arccos(np.clip(1.0 - 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2), -1.0, 1.0))
    nonfoot = np.flatnonzero(np.any(trace["force_n"][:, 0, 2:] > 1.0, axis=1))
    first = outcome["first_foot_contact_frame"]
    post = None
    if first is not None and first + 1 < 300:
        start = first + 1
        xyz = pose[start:, :3]
        delta = np.diff(xyz, axis=0)
        velocity = trace["root_velocity_world"][start:, 0]
        torque = trace["torque_nm"][start:, 0].reshape(-1, 29)
        joint_v = trace["joint_velocity_rad_s"][start:, 0]
        acceleration = np.diff(joint_v, axis=0) / 0.02
        # No unobserved tail and no claimed dense body/contact reconstruction.
        post = dict(
            start_frame=start,
            observed_control_frames=300 - start,
            remaining_task_duration_sec=(300 - start) * 0.02,
            sampled_root_time_span_sec=(299 - start) * 0.02,
            sampled_root_displacement_world_m=(xyz[-1] - xyz[0]).tolist(),
            sampled_root_negative_world_x_travel_m=float(np.maximum(-delta[:, 0], 0).sum()),
            sampled_root_horizontal_path_m=float(np.linalg.norm(delta[:, :2], axis=1).sum()),
            maximum_root_tilt_deg=float(np.rad2deg(tilt[start:]).max()),
            maximum_world_angular_speed_rad_s=float(np.linalg.norm(velocity[:, 3:], axis=1).max()),
            sampled_root_negative_world_x_velocity_frames=int(
                np.count_nonzero(velocity[:, 0] < -0.05)
            ),
            minimum_native_pelvis_z_m=float(trace["pelvis_z_per_substep_m"][start:].min()),
            sampled_joint_acceleration_rms_rad_s2=None
            if not len(acceleration)
            else float(np.sqrt(np.mean(acceleration**2))),
            native_torque_saturation_fraction=float(
                np.mean(np.abs(torque) >= np.asarray(G1_HARD_TORQUE_LIMITS) * (1.0 - 1e-6))
            ),
            native_mean_abs_torque_change_nm=None
            if len(torque) < 2
            else float(np.abs(np.diff(torque, axis=0)).mean()),
        )
    report = dict(
        schema="soccer.rsi.recovery_diagnostics.v1",
        formal_outcome=outcome,
        unsafe_components=dict(
            pelvis_below_0_65_m=bool(np.any(trace["pelvis_z_per_substep_m"] < 0.65)),
            observed_six_channel_nonfoot_ball_contact=bool(len(nonfoot)),
        ),
        first_observed_nonfoot_ball_contact_frame=int(nonfoot[0]) if len(nonfoot) else None,
        post_contact=post,
        new_physical_steps=0,
        new_nn_inferences=0,
        new_optimizer_updates=0,
        formal_task_or_reward_changed=False,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    return report
