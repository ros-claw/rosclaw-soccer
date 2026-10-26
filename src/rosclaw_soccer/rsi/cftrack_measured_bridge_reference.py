"""Experimental SIM measured-state bridge into original CFgen kick suffix."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash, copy_cftrack_policy
from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
from rosclaw_soccer.sim.contracts import hash_json


def condition_measured_bridge_reference(
    player: SharedOmniContactPlayer,
    roots: np.ndarray,
    joints: np.ndarray,
    *,
    expected_model_hash: str,
    ball_prediction_horizon_s: float | None = None,
) -> dict[str, Any]:
    """Install one prequalified sampled bridge and preserve the original kick suffix.

    This is a separate research path: the original cold direct/staged guards are
    unchanged. Rechecks sampled geometry against the actual owned body; neither
    support force nor interpolated or closed-loop physical safety is established.
    """
    import mujoco
    from policy.omnicontact.CFgen_meta1_loco import (
        DEFAULT_JOINT_POS_MJ,
        _append_fk_block,
        _append_turn,
        _append_walk,
        _fk_reference_sequence_from_joints,
    )

    if type(player) is not SharedOmniContactPlayer:
        raise ValueError("owned private CFtrack SIM player required")
    w, p, lo = player.world, player.policy, player.layout
    w._require_current_boundary()
    if (
        compiled_model_hash(w.model) != expected_model_hash
        or player.failed
        or player.pending is not None
        or player.commits != 0
        or player.boundary != w.control_steps
        or player.motor_plan is not None
        or player.measured_bridge_reference_installed
        or not player.measured_history_installed
        or p.counter_step != 0
        or p.reference_source != "CFgen"
        or p.task != "kickball"
        or p.async_stage_thread is not None
        or type(roots) is not np.ndarray
        or roots.shape != (21, 7)
        or roots.dtype.kind != "f"
        or not np.isfinite(roots).all()
        or type(joints) is not np.ndarray
        or joints.shape != (21, 29)
        or joints.dtype.kind != "f"
        or not np.isfinite(joints).all()
    ):
        raise ValueError(
            "pinned actual body and one installed-history pre-inference bridge required"
        )
    actual_q, actual_v = lo.canonical(w.data)
    if (
        not np.array_equal(roots[0], actual_q[:7])
        or not np.array_equal(joints[0], actual_q[7:36])
        or not np.array_equal(p.state_cmd.obj_pos, w.data.xpos[lo.ball])
        or not np.allclose(np.linalg.norm(roots[:, 3:], axis=1), 1, rtol=0, atol=1e-6)
        or not np.array_equal(roots[:, 3:], np.broadcast_to(roots[0, 3:], (21, 4)))
        or np.max(np.abs(joints[-1] - DEFAULT_JOINT_POS_MJ)) > 0.02
        or np.max(np.abs(np.diff(joints, axis=0))) > 0.12
        or np.max(np.abs(np.diff(roots[:, 2]))) > 0.006
        or np.max(np.abs((joints[1] - joints[0]) / 0.02 - actual_v[6:35])) > 2.0
        or np.any(joints < w.model.jnt_range[lo.joints, 0])
        or np.any(joints > w.model.jnt_range[lo.joints, 1])
    ):
        raise ValueError("actual initial state and unchanged sampled bridge bounds required")
    goal_direction = np.asarray(p.goal_pos[:2], np.float64) - w.data.xpos[lo.ball, :2]
    if np.linalg.norm(goal_direction) < 1e-6:
        raise ValueError("nondegenerate measured goal direction required")
    goal_direction /= np.linalg.norm(goal_direction)
    if (roots[-1, :2] - roots[0, :2]) @ goal_direction < 0:
        raise ValueError("bridge terminal cannot move backward along the goal axis")
    floor = int(w.model.geom("floor").id)
    feet = [
        [int(w.model.geom(f"a/{side}_foot{i}_collision").id) for i in range(1, 8)]
        for side in ("left", "right")
    ]
    foot_set = {g for foot in feet for g in foot}
    ankle_ids = [int(w.model.body(f"a/{side}_ankle_roll_link").id) for side in ("left", "right")]
    data = mujoco.MjData(w.model)
    data.qpos[lo.bq : lo.bq + 7] = [100, 100, 0.11, 1, 0, 0, 0]
    gaps, ankles, nonfoot = [], [], []
    for root, joint in zip(roots, joints, strict=True):
        data.qpos[lo.rq : lo.rq + 7], data.qpos[lo.qi] = root, joint
        mujoco.mj_forward(w.model, data)
        gaps.append(
            [
                min(float(mujoco.mj_geomDistance(w.model, data, floor, g, 1.0, None)) for g in foot)
                for foot in feet
            ]
        )
        ankles.append(data.xpos[ankle_ids, :2].copy())
        contacts = [
            float(c.dist)
            for c in data.contact
            if floor in (int(c.geom[0]), int(c.geom[1]))
            and not ({int(c.geom[0]), int(c.geom[1])} & foot_set)
        ]
        nonfoot.append(min(contacts, default=0.0))
    gaps, ankles = np.asarray(gaps), np.asarray(ankles)
    support = (gaps[0] >= -0.002) & (gaps[0] <= 0.003)
    if (
        support.tolist() != [False, True]
        or np.min(gaps) < -0.002
        or np.max(np.abs(gaps[:, support])) > 0.01
        or np.min(nonfoot) < -0.002
        or np.max(np.linalg.norm((ankles - ankles[0])[:, support], axis=2)) > 0.02
        or np.max(np.linalg.norm(np.diff(ankles[:, support], axis=0), axis=2)) > 0.006
        or data.time != 0
    ):
        raise ValueError("sampled support proxy, floor or ankle-continuity bounds failed")
    before_world = snapshot(w.data)
    before_ball = w.data.xpos[lo.ball].copy(), w.data.xquat[lo.ball].copy()
    before_policy = cftrack_state_hash(p)
    candidate = copy_cftrack_policy(p)
    planned_ball = p.state_cmd.obj_pos.copy()
    prediction_receipt = None
    if ball_prediction_horizon_s is not None:
        from policy.omnicontact.CFgen_reference import plan_cfgen_reference

        from rosclaw_soccer.rsi.cftrack_reference_ball_prediction import predict_ball_reference

        planned_ball, prediction_receipt = predict_ball_reference(
            np.asarray(p.state_cmd.obj_pos).copy(),
            w.data.qvel[lo.bv : lo.bv + 3].copy(),
            horizon_s=ball_prediction_horizon_s,
        )
        # Only the private planning copy sees the predicted position. The
        # original policy and live ball observations stay measured.
        candidate.state_cmd.ball_pos = planned_ball.copy()
        candidate.state_cmd.obj_pos = planned_ball.copy()
        fk = candidate.kinematics.forward(
            candidate.state_cmd.q, candidate.state_cmd.base_pos, candidate.state_cmd.base_quat
        )
        plan_cfgen_reference(candidate, fk)
    phase = candidate.ref_phase
    tail = np.flatnonzero(phase >= 13)
    if not len(tail) or tail[0] == 0 or not np.array_equal(tail, np.arange(tail[0], len(phase))):
        raise ValueError("complete original kick and recovery suffix required")
    start = int(tail[0])
    generator = candidate.traj_generator
    stance, goal_quat = (
        candidate.ref_base_pos[start - 1].copy(),
        candidate.ref_base_quat[start - 1].copy(),
    )
    if (stance[:2] - roots[-1, :2]) @ goal_direction < 0:
        raise ValueError("direct walking reference must not retreat along goal axis")
    builder = generator._new_builder()
    refs = _fk_reference_sequence_from_joints(roots[:, :3], roots[:, 3:], joints)
    _append_fk_block(
        builder,
        12,
        fk_refs=refs,
        object_pos=planned_ball.copy(),
        object_quat=p.state_cmd.obj_quat.copy(),
        contact=np.tile(generator._contact0, (21, 1)),
    )
    _append_turn(
        builder,
        12,
        pelvis_pos=roots[-1, :3],
        torso_pos=None,
        yaw_start=roots[-1, 3:],
        yaw_target=goal_quat,
        step_angular=float(generator.step_angular),
        object_pos=planned_ball.copy(),
        object_quat=p.state_cmd.obj_quat.copy(),
        dof_pos=DEFAULT_JOINT_POS_MJ,
        contact=generator._contact0,
        keep_pelvis_z=True,
    )
    _append_walk(
        builder,
        12,
        pelvis_start=roots[-1, :3],
        pelvis_target=stance,
        torso_start=None,
        torso_target=None,
        yaw=goal_quat,
        step_linear=float(generator.step_linear),
        object_pos=planned_ball.copy(),
        object_quat=p.state_cmd.obj_quat.copy(),
        dof_pos=DEFAULT_JOINT_POS_MJ,
        contact=generator._contact0,
        keep_pelvis_z=True,
    )
    prefix = builder.finalize()
    replacements = {}
    for key, value in prefix.items():
        old = getattr(candidate, key)
        if (
            type(old) is not np.ndarray
            or old.shape[0] != len(phase)
            or old.shape[1:] != value.shape[1:]
            or old.dtype != value.dtype
            or not np.isfinite(old).all()
            or not np.isfinite(value).all()
        ):
            raise ValueError("complete finite compatible source reference fields required")
        replacements[key] = np.concatenate([value, old[start:]], axis=0)
    if "ref_phase" not in replacements or "ref_contact" not in replacements:
        raise ValueError("original phase and contact reference required")
    if cftrack_state_hash(p) != before_policy:
        raise ValueError("scratch bridge construction changed actual policy state")
    after_world = snapshot(w.data)
    if (
        any(not np.array_equal(before_world[k], after_world[k]) for k in before_world)
        or not np.array_equal(before_ball[0], w.data.xpos[lo.ball])
        or not np.array_equal(before_ball[1], w.data.xquat[lo.ball])
    ):
        raise ValueError("scratch bridge construction changed actual physical state")
    original_hash = hash_json({k: getattr(p, k).tolist() for k in replacements})
    new_hash = hash_json({k: v.tolist() for k, v in replacements.items()})
    previous = {key: getattr(p, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(p, key, value)
        player.measured_bridge_reference_installed = True
    except BaseException:
        for key, value in previous.items():
            setattr(p, key, value)
        player.measured_bridge_reference_installed = False
        raise
    receipt = dict(
        schema="soccer.cftrack.measured_bridge_reference.v1",
        original_reference_hash=original_hash,
        conditioned_reference_hash=new_hash,
        bridge_frames=21,
        prefix_frames=len(prefix["ref_phase"]),
        original_swing_start=start,
        original_kick_recovery_suffix_unchanged=True,
        actual_ball_reference_static=prediction_receipt is None,
        support_inferred_from_geometry_not_force=True,
        sampled_geometry_not_dynamics=True,
        original_direct_and_staged_guards_unchanged=True,
        physical_state_modified=False,
        reference_qualified_for_motion=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    if prediction_receipt is not None:
        receipt["ball_prediction"] = prediction_receipt
        receipt["kick_suffix_from_predicted_ball_replan"] = True
    return receipt
