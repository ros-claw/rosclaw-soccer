"""A measured-start direct approach for an already aligned kick-only CFtrack skill.

No training, inference or physical state writes. Existing external primitives
are called from the caller-pinned SIM source; nothing is copied or deployed.
"""

from typing import Any

import numpy as np


def condition_direct_kick_approach(policy: Any) -> dict[str, Any]:
    from policy.omnicontact.CFgen_meta1_loco import (  # type: ignore[import-not-found]
        DEFAULT_JOINT_POS_MJ,
        _append_loco_walk,
    )

    from rosclaw_soccer.sim.contracts import hash_json

    if (
        policy.task != "kickball"
        or policy.reference_source != "CFgen"
        or policy.counter_step != 0
        or policy.async_stage_thread is not None
        or np.any(policy.obs_history_buffer)
        or np.any(policy.action)
    ):
        raise ValueError("one cold pre-inference private kick reference required")
    s = policy.state_cmd
    direction = np.asarray(policy.goal_pos[:2] - s.ball_pos[:2], np.float64)
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        raise ValueError("nondegenerate goal direction required")
    direction /= length
    left = np.array([-direction[1], direction[0]])
    relative = s.ball_pos[:2] - s.base_pos[:2]
    qw, qx, qy, qz = s.base_quat
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    desired_yaw = np.arctan2(direction[1], direction[0])
    heading_error = float(abs(np.arctan2(np.sin(yaw - desired_yaw), np.cos(yaw - desired_yaw))))
    if (
        heading_error > 0.2
        or not 0.3 <= float(relative @ direction) <= 1.5
        or abs(float(relative @ left)) > 0.2
        or abs(s.q - DEFAULT_JOINT_POS_MJ).max() > 1e-6
    ):
        raise ValueError("direct approach only qualified for aligned default-pose near-ball starts")
    phase = policy.ref_phase
    tail = np.flatnonzero(phase >= 13)
    if not len(tail) or tail[0] == 0 or not np.array_equal(tail, np.arange(tail[0], len(phase))):
        raise ValueError("complete original kick/recovery suffix required")
    start = int(tail[0])
    # The existing reference is generated from this current measured start.
    # Reuse its final stance and ALL of its swing/recovery suffix, but do not
    # first retreat past the current body to the distant staging waypoint.
    generator = policy.traj_generator
    builder = generator._new_builder()
    _append_loco_walk(
        builder,
        12,
        pelvis_start=s.base_pos.copy(),
        pelvis_target=policy.ref_base_pos[start - 1].copy(),
        yaw=policy.ref_base_quat[start - 1].copy(),
        step_linear=float(generator.step_linear),
        object_pos=s.obj_pos.copy(),
        object_quat=s.obj_quat.copy(),
    )
    prefix = builder.finalize()
    replacements = {}
    for key, value in prefix.items():
        old = getattr(policy, key)
        if (
            type(old) is not np.ndarray
            or old.shape[0] != len(phase)
            or old.shape[1:] != value.shape[1:]
            or old.dtype != value.dtype
            or not np.isfinite(old).all()
            or not np.isfinite(value).all()
        ):
            raise ValueError("complete compatible finite taskspace reference required")
        replacements[key] = np.concatenate([value, old[start:]], axis=0)
    if "ref_phase" not in replacements or "ref_contact" not in replacements:
        raise ValueError("phase/contact reference cannot be dropped")
    old_hash = hash_json({k: getattr(policy, k).tolist() for k in replacements})
    new_hash = hash_json({k: v.tolist() for k, v in replacements.items()})
    for key, value in replacements.items():
        setattr(policy, key, value)
    return dict(
        schema="soccer.cftrack_direct_approach.v1",
        original_reference_hash=old_hash,
        conditioned_reference_hash=new_hash,
        original_frames=len(phase),
        original_swing_start=start,
        direct_approach_frames=len(prefix["ref_phase"]),
        conditioned_frames=len(replacements["ref_phase"]),
        original_kick_and_recovery_suffix_exact=True,
        original_step_linear_m=float(generator.step_linear),
        input_is_current_body_and_goal_not_recorded_future=True,
        neural_weights_modified=False,
        physical_state_modified=False,
        hardware_authorized=False,
    )
