"""Synthetic original-clock readback tests; no G1 success or trained teacher."""

import numpy as np
import pytest

from scripts.rsi_read_zero_feedback import verify_branch


def fixture():
    trace = dict(
        canonical_qpos=np.zeros((300, 1, 43)),
        canonical_qvel=np.zeros((300, 1, 41)),
        pd_requested_torque_nm=np.zeros((300, 1, 10, 29)),
        torque_nm=np.zeros((300, 1, 10, 29)),
        actual_actuator_force_nm=np.zeros((300, 1, 10, 29)),
        pelvis_z_per_substep_m=np.full((300, 1, 10), 0.7),
        force_n=np.zeros((300, 1, 6)),
        foundation_navigation_joint_target_rad=np.zeros((300, 1, 29)),
        taskspace_joint_increment_rad=np.zeros((300, 1, 29)),
        motor_delta_rad=np.zeros((300, 1, 12)),
        joint_target_rad=np.zeros((300, 1, 29)),
    )
    arrays = dict(
        end_integration_state=np.zeros(100),
        canonical_qpos_after_substep=np.zeros((20, 43)),
        canonical_qvel_after_substep=np.zeros((20, 41)),
        requested_pd_torque_nm=np.zeros((20, 29)),
        applied_control_nm=np.zeros((20, 29)),
        actual_actuator_force_nm=np.zeros((20, 29)),
        pelvis_height_solver_epoch_m=np.full(20, 0.7),
        robot_force_on_ball_world_n=np.zeros((20, 3)),
        first_control_peak_force_n=np.zeros(6),
    )
    for name, width in dict(
        navigation_target_rad=29,
        taskspace_increment_rad=29,
        motor_delta_rad=12,
        joint_target_rad=29,
        canonical_qpos=43,
        canonical_qvel=41,
    ).items():
        arrays["next_" + name] = np.zeros(width)
    for name, width in dict(
        encoder_features=640,
        latent_token=64,
        decoder_input=994,
        raw_action_isaac=29,
        joint_target_mujoco_rad=29,
    ).items():
        arrays["neural_next_" + name] = np.zeros((1, width), dtype=np.float32)
        trace["foundation_neural_" + name] = np.zeros((300, 1, width), dtype=np.float32)
    dense = dict(
        canonical_qpos_before_substep=np.zeros((3000, 43)),
        canonical_qvel_after_substep=np.zeros((3000, 41)),
        robot_contact_force_on_ball_world_n=np.zeros((3000, 3)),
    )
    return arrays, trace, dense


def test_whole_zero_branch_original_clock():
    verify_branch(*fixture(), 30)


@pytest.mark.parametrize(
    "bad",
    [
        "dense_qpos",
        "dense_qvel",
        "dense_force",
        "native_pd",
        "native_ctrl",
        "native_force",
        "native_pelvis",
        "peaks",
        "next_ik",
        "next_residual",
        "neural",
        "signed_zero",
        "end_nan",
        "short",
    ],
)
def test_rejects_tampering_in_every_stage_of_feedback(bad):
    a, t, d = fixture()
    if bad == "dense_qpos":
        d["canonical_qpos_before_substep"][301, 0] = 1
    elif bad == "dense_qvel":
        d["canonical_qvel_after_substep"][300, 0] = 1
    elif bad == "dense_force":
        d["robot_contact_force_on_ball_world_n"][319, 0] = 1
    elif bad == "native_pd":
        t["pd_requested_torque_nm"][31, 0, 9, 0] = 1
    elif bad == "native_ctrl":
        t["torque_nm"][31, 0, 9, 0] = 1
    elif bad == "native_force":
        t["actual_actuator_force_nm"][31, 0, 9, 0] = 1
    elif bad == "native_pelvis":
        t["pelvis_z_per_substep_m"][31, 0, 9] = 1
    elif bad == "peaks":
        a["first_control_peak_force_n"][0] = 1
    elif bad == "next_ik":
        a["next_taskspace_increment_rad"][0] = 1
    elif bad == "next_residual":
        a["next_motor_delta_rad"][0] = 1
    elif bad == "neural":
        a["neural_next_decoder_input"][0, 64] = 1
    elif bad == "signed_zero":
        a["neural_next_encoder_features"][0, 0] = -0.0
    elif bad == "end_nan":
        a["end_integration_state"][0] = np.nan
    else:
        a["canonical_qpos_after_substep"] = a["canonical_qpos_after_substep"][:19]
    with pytest.raises(ValueError):
        verify_branch(a, t, d, 30)
