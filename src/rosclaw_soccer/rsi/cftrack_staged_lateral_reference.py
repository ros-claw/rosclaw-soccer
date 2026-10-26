"""Plan a bounded lateral pre-positioning reference for a moving ball in SIM.

This is a separate staged reference, not a relaxation of direct-approach
qualification. A caller must explicitly check the actual stage boundary before
allowing any subsequent kick controls. No physical state is changed here.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cftrack_alignment_diagnostic import direct_approach_geometry
from rosclaw_soccer.rsi.cftrack_velocity_intercept import predict_initial_intercept
from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash
from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
from rosclaw_soccer.sim.contracts import hash_json


def lateral_stage_waypoint(
    base: np.ndarray, quaternion: np.ndarray, ball: np.ndarray, goal: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    geometry = direct_approach_geometry(base, quaternion, ball, goal)
    lateral = float(geometry["lateral_offset_m"])
    if (
        not geometry["heading_within_guard"]
        or not geometry["forward_within_guard"]
        or not 0.2 < abs(lateral) <= 0.3
    ):
        raise ValueError("explicit bounded lateral-stage applicability required")
    direction = (goal[:2] - ball[:2]).astype(np.float64)
    direction /= np.linalg.norm(direction)
    left = np.array([-direction[1], direction[0]])
    displacement = lateral - np.copysign(0.15, lateral)
    if abs(displacement) > 0.15:
        raise ValueError("lateral-stage displacement must not exceed 0.15m")
    waypoint = base.astype(np.float64, copy=True)
    waypoint[:2] += displacement * left
    waypoint_geometry = direct_approach_geometry(waypoint, quaternion, ball, goal)
    if (
        not waypoint_geometry["heading_within_guard"]
        or not waypoint_geometry["forward_within_guard"]
        or abs(float(waypoint_geometry["lateral_offset_m"])) > 0.15 + 1e-9
    ):
        raise ValueError("planned waypoint does not enter declared geometry domain")
    return waypoint, dict(
        initial_geometry=geometry,
        planned_waypoint_geometry=waypoint_geometry,
        planned_lateral_displacement_m=float(displacement),
        maximum_lateral_displacement_m=0.15,
        planned_waypoint=waypoint.tolist(),
        waypoint_is_reference_not_measured_body=True,
    )


def condition_staged_lateral_intercept(player: SharedOmniContactPlayer) -> dict[str, Any]:
    from policy.omnicontact.CFgen_meta1_loco import (  # type: ignore[import-not-found]
        DEFAULT_JOINT_POS_MJ,
        _append_loco_walk,
    )
    from policy.omnicontact.CFgen_reference import (  # type: ignore[import-not-found]
        plan_cfgen_reference,
    )

    w, p = player.world, player.policy
    w._require_current_boundary()
    if (
        type(player) is not SharedOmniContactPlayer
        or w.native_steps != 0
        or w.control_steps != 0
        or player.failed
        or player.pending is not None
        or player.motor_plan is not None
        or w._torque_layer_owners
        or p.counter_step != 0
        or p.task != "kickball"
        or p.reference_source != "CFgen"
        or p.async_stage_thread is not None
        or np.any(p.obs_history_buffer)
        or np.any(p.action)
    ):
        raise ValueError("unexecuted private CFgen SIM policy required")
    s = p.state_cmd
    if np.max(abs(s.q - DEFAULT_JOINT_POS_MJ)) > 1e-6:
        raise ValueError("measured default-pose initial body required")
    ball = np.asarray(s.ball_pos).copy()
    obj = np.asarray(s.obj_pos).copy()
    if not np.array_equal(ball, w.data.xpos[player.layout.ball]) or not np.array_equal(ball, obj):
        raise ValueError("actual observed initial ball required")
    velocity = w.data.qvel[player.layout.bv : player.layout.bv + 3].copy()
    prediction, forecast = predict_initial_intercept(ball, velocity, np.asarray(p.goal_pos))
    waypoint, geometry = lateral_stage_waypoint(
        np.asarray(s.base_pos), np.asarray(s.base_quat), prediction, np.asarray(p.goal_pos)
    )
    old_policy = cftrack_state_hash(p)
    before_q, before_v, before_time = w.data.qpos.copy(), w.data.qvel.copy(), float(w.data.time)
    try:
        s.ball_pos, s.obj_pos = prediction.copy(), prediction.copy()
        fk = p.kinematics.forward(s.q, s.base_pos, s.base_quat)
        plan_cfgen_reference(p, fk)
        phase = p.ref_phase
        tail = np.flatnonzero(phase >= 13)
        if (
            not len(tail)
            or tail[0] == 0
            or not np.array_equal(tail, np.arange(tail[0], len(phase)))
        ):
            raise ValueError("complete original swing and recovery suffix required")
        start = int(tail[0])
        generator = p.traj_generator
        builder = generator._new_builder()
        yaw = p.ref_base_quat[start - 1].copy()
        first = _append_loco_walk(
            builder,
            12,
            pelvis_start=s.base_pos.copy(),
            pelvis_target=waypoint.copy(),
            yaw=yaw,
            step_linear=0.005,
            object_pos=s.obj_pos.copy(),
            object_quat=s.obj_quat.copy(),
        )
        stage_controls = len(first["base_p"])
        if not 2 <= stage_controls <= 100:
            raise ValueError("bounded slow lateral stage length required")
        _append_loco_walk(
            builder,
            12,
            pelvis_start=waypoint.copy(),
            pelvis_target=p.ref_base_pos[start - 1].copy(),
            yaw=yaw,
            step_linear=float(generator.step_linear),
            object_pos=s.obj_pos.copy(),
            object_quat=s.obj_quat.copy(),
        )
        prefix = builder.finalize()
        replacement = {}
        for key, value in prefix.items():
            old = getattr(p, key)
            if (
                type(old) is not np.ndarray
                or old.shape[0] != len(phase)
                or old.shape[1:] != value.shape[1:]
                or old.dtype != value.dtype
                or not np.isfinite(old).all()
                or not np.isfinite(value).all()
            ):
                raise ValueError("complete compatible finite staged reference required")
            replacement[key] = np.concatenate([value, old[start:]], axis=0)
        if "ref_phase" not in replacement or "ref_contact" not in replacement:
            raise ValueError("phase and contact references are required")
        original_reference_hash = hash_json({k: getattr(p, k).tolist() for k in replacement})
        new_reference_hash = hash_json({k: v.tolist() for k, v in replacement.items()})
        for key, value in replacement.items():
            setattr(p, key, value)
    except BaseException:
        player.failed = True
        raise
    finally:
        s.ball_pos, s.obj_pos = ball, obj
    try:
        w._require_current_boundary()
        if (
            not np.array_equal(w.data.qpos, before_q)
            or not np.array_equal(w.data.qvel, before_v)
            or float(w.data.time) != before_time
            or p.counter_step != 0
            or not np.array_equal(s.ball_pos, ball)
            or not np.array_equal(s.obj_pos, obj)
        ):
            raise ValueError("measured body, ball observations and clock must remain unchanged")
    except BaseException:
        player.failed = True
        raise
    return dict(
        schema="soccer.cftrack.staged_lateral_reference.v1",
        prediction=forecast,
        geometry=geometry,
        stage_controls=stage_controls,
        stage_reference_step_m=0.005,
        original_swing_and_recovery_suffix_exact=True,
        original_reference_hash=original_reference_hash,
        staged_reference_hash=new_reference_hash,
        previous_policy_hash=old_policy,
        conditioned_policy_hash=cftrack_state_hash(p),
        actual_stage_boundary_check_required=True,
        measured_observations_restored_before_inference=True,
        neural_optimizer_updates=0,
        hardware_authorized=False,
    )
