"""Independent80ms raw zero-branch comparison; no simulator/model calls."""

import numpy as np

from scripts.rsi_impact_torque_authority_screen import equal


def verify_early_branch(a, trace, dense, frame):
    if type(frame) is not int or not 30 <= frame <= 294 or frame // 20 != (frame + 3) // 20:
        raise ValueError("registered no-replan80ms window required")
    end = a["end_integration_state"]
    if (
        end.ndim != 1
        or not 41 <= len(end) <= 65536
        or end.dtype != np.float64
        or not np.isfinite(end).all()
    ):
        raise ValueError("whole finite80ms end state required")
    for key, shape in (
        ("canonical_qpos_after_substep", (40, 43)),
        ("canonical_qvel_after_substep", (40, 41)),
        ("requested_pd_torque_nm", (40, 29)),
        ("requested_offset_nm", (40, 29)),
        ("applied_control_nm", (40, 29)),
        ("actual_actuator_force_nm", (40, 29)),
        ("pelvis_height_solver_epoch_m", (40,)),
        ("robot_force_on_ball_world_n", (40, 3)),
        ("completed_contact_peak_force_n", (40, 6)),
        ("foot_contact_peak_force_n_after_substep", (40, 2)),
        ("completed_foot_ground_normal_n", (40, 2)),
    ):
        v = a[key]
        if v.shape != shape or v.dtype != np.float64 or not np.isfinite(v).all():
            raise ValueError("whole finite80ms branch arrays required: " + key)
    nonfoot = a["nonfoot_robot_contact_over1n_after_substep"]
    if nonfoot.shape != (40,) or nonfoot.dtype != np.bool_:
        raise ValueError("typed whole80ms nonfoot diagnostics required")
    if np.any(a["requested_offset_nm"]):
        raise ValueError("zero calibration cannot contain nonzero actions")
    for k in (
        "completed_contact_peak_force_n",
        "foot_contact_peak_force_n_after_substep",
        "completed_foot_ground_normal_n",
    ):
        if np.any(a[k] < 0):
            raise ValueError("nonnegative observed contact magnitudes required")
    start = frame * 10
    for name, original, lo, hi in (
        ("canonical_qpos_after_substep", "canonical_qpos_before_substep", start + 1, start + 41),
        ("canonical_qvel_after_substep", "canonical_qvel_after_substep", start, start + 40),
        ("robot_force_on_ball_world_n", "robot_contact_force_on_ball_world_n", start, start + 40),
    ):
        equal(a[name], dense[original][lo:hi], "all40 dense " + name)
    for name, original, shape in (
        ("requested_pd_torque_nm", "pd_requested_torque_nm", (40, 29)),
        ("applied_control_nm", "torque_nm", (40, 29)),
        ("actual_actuator_force_nm", "actual_actuator_force_nm", (40, 29)),
        ("pelvis_height_solver_epoch_m", "pelvis_z_per_substep_m", (40,)),
    ):
        equal(a[name], trace[original][frame : frame + 4, 0].reshape(shape), "native " + name)
    equal(
        a["completed_contact_peak_force_n"][[9, 19, 29, 39]],
        trace["force_n"][frame : frame + 4, 0],
        "all four original causal contact-force bins",
    )
    for j in range(1, 4):
        for name, original, size in (
            ("navigation_target_rad", "foundation_navigation_joint_target_rad", 29),
            ("taskspace_increment_rad", "taskspace_joint_increment_rad", 29),
            ("motor_delta_rad", "motor_delta_rad", 12),
            ("joint_target_rad", "joint_target_rad", 29),
            ("canonical_qpos", "canonical_qpos", 43),
            ("canonical_qvel", "canonical_qvel", 41),
        ):
            v = a[f"next_{j}_" + name]
            if v.shape != (size,) or v.dtype != np.float64 or not np.isfinite(v).all():
                raise ValueError("whole finite next target/body required")
            equal(v, trace[original][frame + j, 0], f"all three next{j} " + name)
        for name, width in (
            ("encoder_features", 640),
            ("latent_token", 64),
            ("decoder_input", 994),
            ("raw_action_isaac", 29),
            ("joint_target_mujoco_rad", 29),
        ):
            v = a[f"neural_next_{j}_" + name]
            if (
                v.shape != (1, width)
                or v.dtype != np.float32
                or not np.isfinite(v).all()
                or v.tobytes() != trace["foundation_neural_" + name][frame + j].tobytes()
            ):
                raise ValueError(f"actual neural next{j} bits differ: " + name)
