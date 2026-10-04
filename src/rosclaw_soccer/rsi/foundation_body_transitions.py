"""Causal full-body data preparation, not 29D policy-gradient training.

Recorded foundation outputs are not the final applied motor controls. Training
controls must use actual composed joint targets and measured substep forces.
Only the first T-1 next-state pairs exist; a terminal body state is not invented.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.foundation_observation_capture import (
    PREFIX,
    validate_capture,
    validate_measured_history,
)


def body_transitions(contract: dict[str, Any], trace: dict[str, Any]) -> dict[str, Any]:
    """Owned tensors for predictive dynamics or imitation, never a motor permit.

    A caller must independently verify the source physical and ONNX receipts.
    This function only verifies internal dimensions/history/causal alignment.
    The currently sampled behavior has 12 leg dimensions, not a 29D density.
    """
    q = np.asarray(trace.get("canonical_qpos"))
    if q.ndim != 3 or q.shape[2] != 43 or not 2 <= q.shape[0] <= 20000:
        raise ValueError("complete bounded canonical body transition trace required")
    frames, lanes, _ = q.shape
    validate_capture(contract, trace, frames=frames, lanes=lanes)
    validate_measured_history(trace)
    shapes = {
        "canonical_qvel": (frames, lanes, 41),
        "joint_target_rad": (frames, lanes, 29),
        "actual_actuator_force_nm": (frames, lanes, 10, 29),
        "root_pose_xyzw_m": (frames, lanes, 7),
        "root_velocity_world": (frames, lanes, 6),
        "ball_position_before_step_m": (frames, lanes, 3),
        "ball_linear_velocity_before_step_m_s": (frames, lanes, 3),
    }
    for key, shape in shapes.items():
        value = np.asarray(trace.get(key))
        if value.shape != shape or value.dtype.kind not in "fi" or not np.isfinite(value).all():
            raise ValueError("complete finite actual body/control/ball trace required")
    position = trace["ball_position_before_step_m"] - trace["root_pose_xyzw_m"][:, :, :3]
    velocity = (
        trace["ball_linear_velocity_before_step_m_s"] - trace["root_velocity_world"][:, :, :3]
    )
    observation = np.concatenate(
        (trace[PREFIX + "decoder_input"], position, velocity), axis=2
    ).astype(np.float32)
    if not np.isfinite(observation).all():
        raise ValueError("float32 body observations overflowed")
    arrays = {
        "observation": observation[:-1].copy(),
        "applied_virtual_joint_target_rad": np.asarray(trace["joint_target_rad"][:-1]).copy(),
        "actual_actuator_force_nm": np.asarray(trace["actual_actuator_force_nm"][:-1]).copy(),
        "current_native_body_qvel": np.asarray(trace["canonical_qvel"][:-1, :, :35]).copy(),
        "next_native_body_qvel": np.asarray(trace["canonical_qvel"][1:, :, :35]).copy(),
        "frozen_foundation_action_isaac": np.asarray(
            trace[PREFIX + "raw_action_isaac"][:-1]
        ).copy(),
    }
    for value in arrays.values():
        value.flags.writeable = False
    return {
        "schema": "soccer.rsi.foundation_body_transitions.v1",
        "arrays": arrays,
        "observed_frames": frames,
        "lanes": lanes,
        "transition_pairs": (frames - 1) * lanes,
        "terminal_body_transitions_omitted": lanes,
        "control_dt_s": 0.02,
        "force_substep_dt_s": 0.002,
        "observation_dimensions": 1000,
        "applied_control_dimensions": 29,
        "sampled_plastic_action_dimensions": 12,
        "current_body_state_order": "native_canonical_root_qvel6_then_DDS_joint_qvel29",
        "relative_ball_reference": "cached_world_origin_position_and_inertial_COM_velocity",
        "controls_are_measured_poses": False,
        "controls_are_composed_virtual_PD_targets": True,
        "source_receipts_replayed_by_this_function": False,
        "policy_gradient_ready": False,
        "optimizer_updates": 0,
        "promotion_authorized": False,
        "hardware_authorized": False,
    }
