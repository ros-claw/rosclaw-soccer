"""Experimental owned-history install; never changes physical state or grants motion."""

import hashlib
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cftrack_action_history import describe_previous_target_encoding
from rosclaw_soccer.rsi.cftrack_history_observation import assemble_current_ball_observation
from rosclaw_soccer.rsi.cftrack_live_history import CFtrackLiveHistory
from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash, copy_cftrack_policy
from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer


def install_owned_cftrack_history(
    player: SharedOmniContactPlayer, history: CFtrackLiveHistory
) -> dict[str, Any]:
    """Install four prior observations and equivalent prior target at an actual boundary.

    On the next source-policy inference its normal roll appends the currently
    measured fifth observation. Past source actions are encoded from actual PD
    targets and are *not* claimed to be the old policy's original outputs.
    This validates input provenance only, not the reference, gains, dynamics,
    task success, or permission to execute the policy.
    """
    if type(player) is not SharedOmniContactPlayer or type(history) is not CFtrackLiveHistory:
        raise ValueError("owned private CFtrack player and live history collector required")
    w, p = player.world, player.policy
    w._require_current_boundary()
    if (
        player.failed
        or player.pending is not None
        or player.commits != 0
        or player.measured_history_installed
        or player.boundary != w.control_steps
        or player.motor_plan is not None
        or not history.belongs_to(w, player.prefix)
        or p.counter_step != 0
        or p.reference_source != "CFgen"
        or p.obs_history_buffer.shape != (5, 141)
        or p.obs_history_buffer.dtype != np.float32
        or p.action.shape != (29,)
        or p.action.dtype != np.float32
        or np.any(p.obs_history_buffer)
        or np.any(p.action)
    ):
        raise ValueError("one fresh owner-bound CFtrack pre-inference boundary required")
    samples = history.at_current_boundary()
    if tuple(s.control for s in samples) != tuple(range(w.control_steps - 4, w.control_steps + 1)):
        raise ValueError("five chronological source controls ending at current boundary required")
    last = samples[-1]
    if (
        not np.array_equal(p.state_cmd.q, np.asarray(last.canonical_qpos)[7:36])
        or not np.array_equal(p.state_cmd.dq, np.asarray(last.canonical_qvel)[6:35])
        or not np.array_equal(p.state_cmd.lin_vel, np.asarray(last.canonical_qvel)[:3])
        or not np.array_equal(p.state_cmd.ang_vel, np.asarray(last.canonical_qvel)[3:6])
        or not np.array_equal(p.state_cmd.base_pos, np.asarray(last.canonical_qpos)[:3])
        or not np.array_equal(p.state_cmd.base_quat, np.asarray(last.canonical_qpos)[3:7])
        or not np.array_equal(p.state_cmd.obj_pos, last.cached_ball_position)
        or not np.array_equal(p.state_cmd.obj_quat, last.cached_ball_quaternion)
        or not np.array_equal(p.state_cmd.ball_pos, last.cached_ball_position)
        or not np.array_equal(p.state_cmd.ball_quat, last.cached_ball_quaternion)
    ):
        raise ValueError("entered policy must still observe the exact live boundary")
    before = snapshot(w.data)
    before_ball = w.data.xpos[player.layout.ball].copy(), w.data.xquat[player.layout.ball].copy()
    fingerprint = cftrack_state_hash(p)
    scratch = copy_cftrack_policy(p)
    rows, implied, errors = [], [], []
    for sample in samples:
        q, v = np.asarray(sample.canonical_qpos), np.asarray(sample.canonical_qvel)
        s = scratch.state_cmd
        s.q, s.dq = q[7:36].copy(), v[6:35].copy()
        s.base_pos, s.base_quat = q[:3].copy(), q[3:7].copy()
        s.ang_vel = v[3:6].copy()
        qw, qx, qy, qz = s.base_quat
        s.gravity_ori = np.array(
            [2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)],
            np.float32,
        )
        s.obj_pos = np.asarray(sample.cached_ball_position).copy()
        s.obj_quat = np.asarray(sample.cached_ball_quaternion).copy()
        scratch._get_fk_info()
        encoding = describe_previous_target_encoding(
            np.asarray(sample.previous_executed_target),
            scratch.default_angles_lab,
            scratch.action_scale_lab,
            scratch.mj2lab,
            scratch.lab2mj,
        )
        scratch.action = encoding["implied_action_lab"].copy()
        implied.append(scratch.action.copy())
        errors.append(encoding["maximum_target_roundtrip_error_rad"])
        rows.append(assemble_current_ball_observation(scratch))
    if cftrack_state_hash(p) != fingerprint:
        raise ValueError("scratch feature construction changed the private source policy")
    after = snapshot(w.data)
    if (
        any(not np.array_equal(before[k], after[k]) for k in before)
        or not np.array_equal(before_ball[0], w.data.xpos[player.layout.ball])
        or not np.array_equal(before_ball[1], w.data.xquat[player.layout.ball])
    ):
        raise ValueError("scratch feature construction changed live physical state or ball cache")
    history.at_current_boundary()  # Recheck that no real control advanced during construction.
    rows_array = np.stack(rows)
    old_history, old_action = p.obs_history_buffer.copy(), p.action.copy()
    try:
        p.obs_history_buffer[1:] = rows_array[:4]
        p.action = implied[-1].copy()
        player.measured_history_installed = True
    except BaseException:
        p.obs_history_buffer[:] = old_history
        p.action = old_action
        player.measured_history_installed = False
        raise
    return dict(
        schema="soccer.cftrack.owned_measured_history.v1",
        sample_controls=[s.control for s in samples],
        history_rows=4,
        current_frame_appended_only_by_next_inference=True,
        previous_actions_are_implied_from_executed_targets=True,
        maximum_target_roundtrip_error_rad=max(errors),
        observed_history_sha256="sha256:" + hashlib.sha256(rows_array.tobytes()).hexdigest(),
        physical_state_and_ball_cache_unchanged=True,
        policy_reference_qualified=False,
        task_success_proven=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
