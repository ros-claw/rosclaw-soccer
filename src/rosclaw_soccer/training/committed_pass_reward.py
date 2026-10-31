"""Fail-closed immutable-target correction for simulation pass shaping.

This is an explicit opt-in experimental adapter. It does not alter legacy PPO
rewards, authorize a policy update, or turn contact into evidence of success.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.role_self_model import TacticalIntent
from rosclaw_soccer.training.pass_impulse_reward import sampled_pass_impulse_reward


def committed_pass_shaping_delta(
    trace: dict[str, Any], ids: tuple[str, ...]
) -> NDArray[np.float64]:
    """Replace only per-contact shaping direction with the pre-physics commitment.

    A scored contact must bind the same source and receiver to a finite target
    captured before the control frame. Missing or inconsistent evidence fails
    closed, rather than silently using a later moving navigation target.
    """
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError("eight unique team agents required")
    time = np.asarray(trace["time"], dtype=np.float64)
    count = len(time)
    if time.shape != (count,) or not np.isfinite(time).all():
        raise ValueError("finite frame times required")
    names = (
        "pass_entry_time_sec",
        "pass_entry_source_code",
        "pass_entry_receiver_code",
        "pass_entry_created_sec",
        "pass_entry_target_valid",
    )
    try:
        entry_time, source_entry, receiver_entry, created, target_valid = (
            np.asarray(trace[name]) for name in names
        )
        fixed_target = np.asarray(trace["pass_entry_target_xy_m"], dtype=np.float64)
        entry_velocity = np.asarray(trace["pass_entry_ball_velocity_xy_mps"], dtype=np.float64)
        entry_ball = np.asarray(trace["pass_entry_ball_xy_m"], dtype=np.float64)
        precontrol = np.asarray(trace["team_native_pre_control_qpos"], dtype=np.float64)
        ball = np.asarray(trace["ball_pose"], dtype=np.float64)[:, :2]
        before = np.asarray(trace["ball_control_entry_velocity"], dtype=np.float64)[:, :2]
        after = np.asarray(trace["ball_velocity"], dtype=np.float64)[:, :2]
        source = np.asarray(trace["pass_source_agent_code"])
        receiver = np.asarray(trace["pass_target_agent_code"])
        contact = np.asarray(trace["ball_contact_agent_code"])
        effector = np.asarray(trace["ball_contact_effector_code"])
        force = np.asarray(trace["ball_contact_force_n"], dtype=np.float64)
        nonfoot = np.asarray(trace["ball_nonfoot_contact_force_n"], dtype=np.float64)
    except KeyError as exc:
        raise ValueError("committed pass telemetry is missing") from exc
    if (
        any(
            x.shape != (count,)
            for x in (
                entry_time,
                source_entry,
                receiver_entry,
                created,
                target_valid,
                source,
                receiver,
                contact,
                effector,
                force,
                nonfoot,
            )
        )
        or target_valid.dtype != np.bool_
        or any(x.dtype.kind not in "iu" for x in (source_entry, receiver_entry, source, receiver))
        or any(
            x.shape != (count, 2)
            for x in (fixed_target, entry_velocity, entry_ball, ball, before, after)
        )
        or precontrol.ndim != 3
        or precontrol.shape[:2] != (count, 8)
        or precontrol.shape[2] < 38
        or not all(
            np.isfinite(x).all()
            for x in (
                time,
                entry_time,
                created,
                fixed_target,
                entry_velocity,
                entry_ball,
                precontrol,
                ball,
                before,
                after,
                force,
                nonfoot,
            )
        )
        or np.any(force < 0)
        or np.any(nonfoot < 0)
    ):
        raise ValueError("invalid committed pass telemetry")
    result = np.zeros((count, 8), dtype=np.float64)
    for i, agent in enumerate(ids):
        key = agent.replace(".", "_")
        try:
            target = np.asarray(trace[key + "_target_position"], dtype=np.float64)[:, :2]
            intent = np.asarray(trace[key + "_intent_code"])
        except KeyError as exc:
            raise ValueError("missing sender intent or target") from exc
        if target.shape != (count, 2) or intent.shape != (count,) or not np.isfinite(target).all():
            raise ValueError("invalid sender intent or target")
        teammates = tuple(
            j + 1
            for j, other in enumerate(ids)
            if j != i and other.partition(".")[0] == agent.partition(".")[0]
        )
        valid = (
            (intent == list(TacticalIntent).index(TacticalIntent.PASS))
            & (contact == i + 1)
            & np.isin(effector, (1, 2))
            & (force > 1.0)
            & (source == i + 1)
            & np.isin(receiver, teammates)
            & (nonfoot == 0.0)
        )
        if np.any(
            valid
            & (
                ~target_valid
                | (source_entry != source)
                | (receiver_entry != receiver)
                | (created < 0)
                | (created > entry_time)
                | (entry_time >= time)
                | ~np.isclose(time - entry_time, 0.02, rtol=0, atol=1e-8)
            )
        ):
            raise ValueError("scored contact lacks a prior matching committed target")
        if not np.array_equal(entry_velocity[valid], before[valid]):
            raise ValueError("committed telemetry is not from the scored control entry")
        if not np.array_equal(entry_ball[valid], precontrol[valid, i, 36:38]):
            raise ValueError(
                "committed ball position does not match the sender's pre-control state"
            )
        old = sampled_pass_impulse_reward(
            ball_xy=ball,
            target_xy=target,
            velocity_before_xy=before,
            velocity_after_xy=after,
            valid_contact=valid,
        )
        fixed = sampled_pass_impulse_reward(
            ball_xy=ball,
            target_xy=fixed_target,
            velocity_before_xy=before,
            velocity_after_xy=after,
            valid_contact=valid,
        )
        result[:, i] = fixed - old
    return result


def committed_directed_physical_rewards(
    trace: dict[str, Any], ids: tuple[str, ...], *, gamma: float = 0.99
) -> NDArray[np.float64]:
    """Explicit immutable-target variant of legacy directed pass shaping."""
    from rosclaw_soccer.training.near_ball_residual_ppo import physical_rewards

    return physical_rewards(trace, ids, reward_shaping="directed_pass_credit_v1", gamma=gamma) + (
        committed_pass_shaping_delta(trace, ids)
    )
