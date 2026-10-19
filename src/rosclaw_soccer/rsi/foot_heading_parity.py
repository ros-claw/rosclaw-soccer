"""Strict C3/C2 array identity; no physics execution or policy admission."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.foot_heading_evidence import FIELDS, PREFIX

# Exact original C2 positive-arm ABI. Not merely a count: replacing one
# physical tensor with an arbitrary field must not pass an OFF regression.
SHAPES = {
    "root_pose_xyzw_m": (7,),
    "root_velocity_world": (6,),
    "ball_position_before_step_m": (3,),
    "ball_linear_velocity_before_step_m_s": (3,),
    "foot_geometry_position_before_step_m": (4, 3),
    "joint_position_rad": (29,),
    "joint_velocity_rad_s": (29,),
    "joint_target_rad": (29,),
    "force_n": (6,),
    "ball_position_after_step_m": (3,),
    "torque_nm": (10, 29),
    "pre_motor_joint_target_rad": (29,),
    "motor_delta_rad": (12,),
    "navigation_command": (3,),
    "actual_actuator_force_nm": (10, 29),
    "pelvis_z_per_substep_m": (10,),
    "canonical_qpos": (43,),
    "canonical_qvel": (41,),
    "recurrent_residual_hidden_state": (64,),
    "recurrent_sampling_raw_mean": (12,),
    "recurrent_sampling_latent_action": (12,),
    "recurrent_sampling_conditional_log_probability": (1,),
    "recurrent_sampling_performed": (1,),
    "foundation_neural_encoder_features": (640,),
    "foundation_neural_latent_token": (64,),
    "foundation_neural_decoder_input": (994,),
    "foundation_neural_raw_action_isaac": (29,),
    "foundation_neural_joint_target_mujoco_rad": (29,),
    "ik_planner_pulse_baseline_command": (3,),
    "ik_planner_pulse_requested_residual": (3,),
    "ik_planner_pulse_applied_command": (3,),
    "approach_requested_navigation_command": (3,),
    "foundation_navigation_joint_target_rad": (29,),
    "taskspace_joint_increment_rad": (29,),
    "motor_pre_projection_requested_rad": (12,),
    "pd_requested_torque_nm": (10, 29),
}


def original_arrays(value: dict[str, Any]) -> dict[str, np.ndarray]:
    extra = {k for k in value if k.startswith(PREFIX)}
    if extra and extra != {PREFIX + k for k in FIELDS}:
        raise ValueError("exact complete heading capture or no heading capture required")
    for key, width in FIELDS.items():
        if extra:
            a = value[PREFIX + key]
            if (
                not isinstance(a, np.ndarray)
                or a.shape != (300, 1, width)
                or a.dtype != np.float64
                or not np.isfinite(a).all()
            ):
                raise ValueError("complete finite heading fields required before stripping")
    arrays = {k: v for k, v in value.items() if k not in extra}
    if arrays.keys() != SHAPES.keys():
        raise ValueError("exact original C2 positive36 fields required, no hidden mixed arrays")
    for key, shape in SHAPES.items():
        a = arrays[key]
        dtype = (
            np.bool_
            if key == "recurrent_sampling_performed"
            else np.float32
            if key.startswith("foundation_neural_")
            else np.float64
        )
        if (
            not isinstance(a, np.ndarray)
            or a.shape != (300, 1, *shape)
            or a.dtype != dtype
            or not np.isfinite(a).all()
        ):
            raise ValueError("whole finite H300 original C2 tensor ABI required")
    return arrays


def require_heading_parity(
    trace: dict[str, Any], baseline: dict[str, Any], *, frames: int
) -> dict[str, Any]:
    """Caller authenticates the C2 positive-arm receipt/source, not this helper."""
    if type(frames) is not int or frames not in (30, 300):
        raise ValueError("exact heading OFF300 or active30 prefix required")
    left, right = original_arrays(trace), original_arrays(baseline)
    if any(left[k][:frames].tobytes() != right[k][:frames].tobytes() for k in SHAPES):
        raise ValueError("all original C2 positive-arm array bits must match")
    return dict(
        compared_frames=frames,
        physical_array_count=36,
        original_array_bits_equal=True,
        source_receipt_authenticated_here=False,
        runtime_execution_authorized=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
