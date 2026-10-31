"""On-policy private leg residual PPO in CPU MuJoCo; never a promotion service.

Each collection freezes eight private actors. Updates happen between episodes,
not inside the physics loop. A finite rollout is an episodic training task; its
terminal value is zero. Dense shaping is diagnostic, not a completed-pass claim.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import re
import tempfile
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.growth.owned_ball_contact import OwnedBallContactPolicy
from rosclaw_soccer.growth.pass_failure_feedback import diagnose_passes
from rosclaw_soccer.growth.per_player_motor_trace import per_player_motor_columns
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.active_team_probe import run_probe, validate_probe
from rosclaw_soccer.training.contact_control_profile import ContactControlProfile
from rosclaw_soccer.training.football_reward_shaping import (
    REWARD_SHAPING_MODES,
    joint_safety_penalty,
    terminal_approach_shaping,
)
from rosclaw_soccer.training.four_vs_four_match import build_four_vs_four_fixture
from rosclaw_soccer.training.motor_task_rewards import motor_task_targets
from rosclaw_soccer.training.near_ball_curriculum import (
    TRAIN_OFFSETS,
    RoleRolloutJob,
    collect_role_course,
    examination_courses,
    training_batch,
)
from rosclaw_soccer.training.near_ball_learning_audit import _verify_contact_profile
from rosclaw_soccer.training.near_ball_plasticity import (
    begin_update,
    finish_update,
    private_weight_hashes,
)
from rosclaw_soccer.training.returned_ball_learning import require_returned_live_segment
from rosclaw_soccer.training.role_behavior_anchor import RoleBehaviorAnchor

PHYSICAL_REWARD_CONTRACT = "rosclaw_soccer.physical_reward.force_gate.v2"


def _persist_training_manifest(output: Path, manifest: Mapping[str, Any]) -> None:
    """Publish a complete progress snapshot without truncating the previous one."""
    payload = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output, prefix=".training-", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output / "training.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def physical_rewards(
    trace: dict[str, Any],
    ids: tuple[str, ...],
    *,
    reward_shaping: str = "legacy",
    gamma: float = 0.99,
) -> np.ndarray[Any, Any]:
    """Foot approach + directed contact, minus tilt, impacts and residual effort.

    No reward for a planner merely declaring PASS. Ball motion is rewarded only
    on this player's measured foot-contact frames. Dense contact is capped by
    the control period, so resting on the ball cannot earn an event bonus.
    """
    if reward_shaping == "team_capture_in_play_impact_v1":
        from rosclaw_soccer.training.nonfoot_impact_reward import nonfoot_impact_adjustment

        base = physical_rewards(
            trace, ids, reward_shaping="team_capture_dissipation_in_play_v1", gamma=gamma
        )
        return base + nonfoot_impact_adjustment(trace, ids)
    if reward_shaping == "team_capture_dissipation_in_play_v1":
        from rosclaw_soccer.training.in_play_rewards import in_play_rewards
        from rosclaw_soccer.world.match_boundary import ball_exit_reason

        # Reward the complete existing receiving objective before the match
        # objective. Reject post-boundary samples: zeroing later rewards alone
        # would still let GAE and capture windows use post-terminal values.
        base = physical_rewards(
            trace, ids, reward_shaping="team_capture_dissipation_v1", gamma=gamma
        )
        result = in_play_rewards(base, trace, ids)
        left, right, _half_width, radius, width, height = np.asarray(
            trace["pitch_boundary_geometry"]
        )[0]
        for point in np.asarray(trace["ball_pose"])[:-1, :3]:
            if (
                ball_exit_reason(
                    tuple(float(v) for v in point),
                    left_x=left,
                    right_x=right,
                    radius=radius,
                    goal_width=width,
                    goal_height=height,
                )
                is not None
            ):
                raise ValueError("in-play PPO rollout must terminate at the first boundary")
        return result
    require_returned_live_segment(trace)
    obs = np.asarray(trace["residual_observations"], dtype=float)
    if reward_shaping not in REWARD_SHAPING_MODES:
        raise ValueError("unknown reward shaping contract")
    count = len(trace["time"])
    dimension = _trace_observation_dimension(trace, count)
    if obs.shape != (count, 8, dimension) or not np.all(np.isfinite(obs)):
        raise ValueError("PPO observations are not finite physical samples")
    rewards = np.zeros((count, 8))
    ball = np.asarray(trace["ball_pose"])[:, :3]
    contact = np.asarray(trace["ball_contact_agent_code"])
    foot = np.isin(trace["ball_contact_effector_code"], [1, 2])
    contact_force = np.asarray(trace["ball_contact_force_n"], dtype=float)
    if (
        contact_force.shape != (count,)
        or not np.isfinite(contact_force).all()
        or np.any(contact_force < 0)
    ):
        raise ValueError("physical reward requires valid nonnegative contact force")
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    for i, agent in enumerate(ids):
        key = agent.replace(".", "_")
        feet = [
            np.asarray(trace[key + suffix])
            for suffix in ("_left_foot_position", "_right_foot_position")
        ]
        before = np.minimum(
            np.linalg.norm(obs[:, i, 38:41], axis=1), np.linalg.norm(obs[:, i, 41:44], axis=1)
        )
        after = np.minimum(*(np.linalg.norm(f - ball, axis=1) for f in feet))
        # Approach gain cannot be maximized merely by dwelling beside the ball.
        rewards[:, i] = 2.0 * (np.exp(-4 * after) - np.exp(-4 * before))
        target = np.asarray(trace[key + "_target_position"])
        if reward_shaping in {"motor_task_contact_v1", "in_play_motor_task_v1"}:
            if "option_target_position_m" not in trace or "option_agent_code" not in trace:
                raise ValueError("motor-task reward requires recorded option targets")
            per_player = per_player_motor_columns(trace, agent, agent_code=i + 1, frames=count)
            target = motor_task_targets(
                target,
                np.asarray(trace["option_target_position_m"])
                if per_player is None
                else per_player[1],
                np.asarray(trace["option_agent_code"]) if per_player is None else per_player[0],
                agent_code=i + 1,
            )
        target = target[:, :2]
        direction = target - ball[:, :2]
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
        toward = (np.asarray(trace["ball_velocity"])[:, :2] * direction).sum(axis=1)
        touching = (contact == i + 1) & foot & (contact_force > 1.0)
        rewards[:, i] += touching * 0.04 * np.clip(toward, -2, 2)
        if reward_shaping in {"pass_impulse_alignment_v1", "directed_pass_credit_v1"}:
            from rosclaw_soccer.growth.role_self_model import TacticalIntent
            from rosclaw_soccer.training.pass_impulse_reward import sampled_pass_impulse_reward

            before_velocity = np.asarray(trace["ball_control_entry_velocity"], dtype=np.float64)
            after_velocity = np.asarray(trace["ball_velocity"], dtype=np.float64)
            sources = np.asarray(trace["pass_source_agent_code"])
            receivers = np.asarray(trace["pass_target_agent_code"])
            nonfoot_force = np.asarray(trace["ball_nonfoot_contact_force_n"], dtype=np.float64)
            intents = np.asarray(trace[key + "_intent_code"])
            if (
                before_velocity.shape != (count, 6)
                or after_velocity.shape != (count, 6)
                or sources.shape != (count,)
                or receivers.shape != (count,)
                or intents.shape != (count,)
                or nonfoot_force.shape != (count,)
                or not np.isfinite(before_velocity).all()
                or not np.isfinite(after_velocity).all()
                or not np.isfinite(nonfoot_force).all()
                or np.any(nonfoot_force < 0)
                or sources.dtype.kind not in "iu"
                or receivers.dtype.kind not in "iu"
                or intents.dtype.kind not in "iu"
                or not np.isin(sources, range(9)).all()
                or not np.isin(receivers, range(9)).all()
                or not np.isin(intents, range(len(TacticalIntent))).all()
            ):
                raise ValueError("pass impulse shaping requires complete typed physical telemetry")
            passing = intents == list(TacticalIntent).index(TacticalIntent.PASS)
            teammates = tuple(
                j + 1
                for j, other in enumerate(ids)
                if j != i and other.partition(".")[0] == agent.partition(".")[0]
            )
            valid = (
                passing
                & touching
                & (sources == i + 1)
                & np.isin(receivers, teammates)
                & (nonfoot_force == 0.0)
            )
            # Avoid rewarding a pre-existing ball velocity or a PASS proposal
            # lacking a matching physical contact and receiver commitment.
            rewards[:, i] -= passing * touching * 0.04 * np.clip(toward, -2, 2)
            rewards[:, i] += sampled_pass_impulse_reward(
                ball_xy=np.asarray(ball[:, :2], dtype=np.float64),
                target_xy=np.asarray(target, dtype=np.float64),
                velocity_before_xy=before_velocity[:, :2],
                velocity_after_xy=after_velocity[:, :2],
                valid_contact=np.asarray(valid, dtype=np.bool_),
            )
        if reward_shaping in {"team_capture_v1", "team_capture_dissipation_v1"}:
            from rosclaw_soccer.growth.role_self_model import TacticalIntent

            receiving = np.isin(
                trace[key + "_intent_code"],
                [
                    list(TacticalIntent).index(x)
                    for x in (
                        TacticalIntent.RECEIVE,
                        TacticalIntent.PRESS,
                        TacticalIntent.INTERCEPT,
                    )
                ],
            )
            rewards[:, i] -= receiving * touching * 0.04 * np.clip(toward, -2, 2)
        if reward_shaping in {"role_receiving_v1", "team_capture_dissipation_v1"}:
            from rosclaw_soccer.growth.role_self_model import TacticalIntent
            from rosclaw_soccer.training.receiving_contact_reward import (
                receiving_contact_adjustment,
            )

            if "ball_control_entry_velocity" not in trace:
                raise ValueError("role receiving requires measured control-entry ball velocity")
            before_velocity = np.asarray(trace["ball_control_entry_velocity"])
            if before_velocity.shape != (count, 6) or not np.isfinite(before_velocity).all():
                raise ValueError("role receiving requires recorded pre-control ball velocity")
            roles = np.asarray(trace[key + "_intent_code"])
            if (
                roles.shape != (count,)
                or roles.dtype.kind not in "iu"
                or not np.isin(roles, range(len(TacticalIntent))).all()
            ):
                raise ValueError("role receiving requires valid actual intent codes")
            rewards[:, i] += receiving_contact_adjustment(
                time=np.asarray(trace["time"]),
                receiving=np.isin(
                    roles,
                    [
                        list(TacticalIntent).index(x)
                        for x in (
                            TacticalIntent.RECEIVE,
                            TacticalIntent.PRESS,
                            TacticalIntent.INTERCEPT,
                        )
                    ],
                ),
                foot_contact=touching,
                owns_ball=np.asarray(trace["possession_agent_code"]) == i + 1,
                speed_before=np.linalg.norm(before_velocity[:, :2], axis=1),
                speed_after=np.linalg.norm(np.asarray(trace["ball_velocity"])[:, :2], axis=1),
                foot_distance=after,
                # Team capture already removes directed credit for receiving;
                # do not remove it a second time in the combined mode.
                directed_reward=(
                    np.zeros(count)
                    if reward_shaping == "team_capture_dissipation_v1"
                    else touching * 0.04 * np.clip(toward, -2, 2)
                ),
            )
        rewards[:, i] -= 0.02 * (nonfoot == i + 1)
        gravity = obs[:, i, :3]
        rewards[:, i] -= 0.015 * np.square(gravity[:, :2]).sum(axis=1)
        height = np.asarray(trace[key + "_pelvis_pose"])[:, 2]
        rewards[:, i] -= 0.2 * (height < 0.65)
        rewards[:, i] -= 0.01 * np.square(trace["residual_applied"][:, i]).sum(axis=1)
        impact = (np.asarray(trace["robot_robot_contact_first_code"]) == i + 1) | (
            np.asarray(trace["robot_robot_contact_second_code"]) == i + 1
        )
        rewards[:, i] -= 0.03 * impact
    # Delayed credit goes to both participants only after a physical reception.
    # Never use the planner's pass handshake alone as a success reward.
    credited = set()
    time = np.asarray(trace["time"])
    launch_relative = "pass_feedback_launch_relative" in trace
    if reward_shaping == "directed_pass_credit_v1" and not launch_relative:
        raise ValueError("directed pass credit requires strict physical handoff telemetry")
    if launch_relative:
        contract = np.asarray(trace["pass_feedback_launch_relative"])
        if contract.shape != time.shape or contract.dtype != np.bool_ or not np.all(contract):
            raise ValueError("physical reward handoff contract is invalid")
    for outcome in diagnose_passes(trace, ids, launch_relative=launch_relative):
        if not outcome["physical_receive_confirmed"]:
            continue
        sender, receiver = ids.index(outcome["sender"]), ids.index(outcome["receiver"])
        first = int(np.searchsorted(time, outcome["foot_contact_time_sec"]))
        if first >= count or contact_force[first] <= 1.0:
            continue
        if reward_shaping == "directed_pass_credit_v1":
            from rosclaw_soccer.training.pass_launch_evidence import inspect_pass_launch

            # Reward attribution uses the most recent completed observation;
            # a successful receiver touch alone cannot invent a pass request.
            # 50 Hz values are conservative proxies, not substep proof.
            prior = int(np.searchsorted(time, float(time[first]), side="left")) - 1
            if prior < 0:
                continue
            receiver_pose = np.asarray(
                trace[ids[receiver].replace(".", "_") + "_pelvis_pose"], dtype=np.float64
            )
            if receiver_pose.shape != (count, 7) or not np.isfinite(receiver_pose).all():
                raise ValueError("directed pass credit requires finite receiver poses")
            measured_velocity = np.asarray(trace["ball_velocity"], dtype=np.float64)
            evidence = inspect_pass_launch(
                control_time_sec=time,
                source_codes=np.asarray(trace["pass_source_agent_code"]),
                receiver_codes=np.asarray(trace["pass_target_agent_code"]),
                sender_code=sender + 1,
                receiver_code=receiver + 1,
                contact_time_sec=float(time[first]),
                ball_xy_m=(float(ball[first, 0]), float(ball[first, 1])),
                ball_velocity_xy_mps=(
                    float(measured_velocity[first, 0]),
                    float(measured_velocity[first, 1]),
                ),
                receiver_position_xy_m=(
                    float(receiver_pose[prior, 0]),
                    float(receiver_pose[prior, 1]),
                ),
            )
            if not evidence.directed_launch_observed:
                continue
        received = np.flatnonzero(
            (contact == receiver + 1)
            & foot
            & (contact_force > 1.0)
            & (time > time[first])
            & (time <= (time[first] if launch_relative else outcome["commitment_time_sec"]) + 3.0)
        )
        if not len(received):
            continue
        end = int(received[0])
        event = (sender, receiver, end)
        if np.linalg.norm(ball[end, :2] - ball[first, :2]) >= 0.5 and event not in credited:
            rewards[end, sender] += 1.0
            rewards[end, receiver] += 1.0
            credited.add(event)
    if not np.all(np.isfinite(rewards)):
        raise ValueError("nonfinite physical reward")
    if reward_shaping in {
        "terminal_potential_v1",
        "contact_safety_v1",
        "motor_task_contact_v1",
        "in_play_motor_task_v1",
        "role_receiving_v1",
        "team_capture_v1",
        "team_capture_dissipation_v1",
        "pass_impulse_alignment_v1",
        "directed_pass_credit_v1",
    }:
        distance = np.minimum(
            np.linalg.norm(obs[:, :, 38:41], axis=2),
            np.linalg.norm(obs[:, :, 41:44], axis=2),
        )
        # Replace only the approach term; event rewards and penalties are unchanged.
        for i, agent in enumerate(ids):
            key = agent.replace(".", "_")
            after = np.minimum(
                *(
                    np.linalg.norm(np.asarray(trace[key + suffix]) - ball, axis=1)
                    for suffix in ("_left_foot_position", "_right_foot_position")
                )
            )
            rewards[:, i] -= 2.0 * (np.exp(-4 * after) - np.exp(-4 * distance[:, i]))
        rewards += terminal_approach_shaping(distance, gamma=gamma)
    if reward_shaping in {
        "contact_safety_v1",
        "motor_task_contact_v1",
        "in_play_motor_task_v1",
        "role_receiving_v1",
        "team_capture_v1",
        "team_capture_dissipation_v1",
        "pass_impulse_alignment_v1",
        "directed_pass_credit_v1",
    }:
        margins = np.stack(
            [np.asarray(trace[a.replace(".", "_") + "_joint_safety_margin_rad"]) for a in ids],
            axis=1,
        )
        if margins.shape != (count, 8, 29):
            raise ValueError("joint margin trace does not match reward observations")
        rewards += joint_safety_penalty(margins)
    if reward_shaping == "in_play_motor_task_v1":
        from rosclaw_soccer.training.in_play_rewards import in_play_rewards

        rewards = in_play_rewards(rewards, trace, ids)
    if reward_shaping in {"team_capture_v1", "team_capture_dissipation_v1"}:
        from rosclaw_soccer.training.team_capture_curriculum import team_capture_trials

        capture_reward, _ = team_capture_trials(trace, agent_ids=ids)
        rewards += capture_reward
    return rewards


def episodic_gae(
    rewards: np.ndarray[Any, Any],
    values: np.ndarray[Any, Any],
    *,
    gamma: float = 0.99,
    trace_decay: float = 0.95,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    if not 0.9 <= gamma < 1 or not 0.9 <= trace_decay < 1:
        raise ValueError("bounded control-rate credit parameters required")
    if rewards.shape != values.shape or not np.all(np.isfinite(rewards + values)):
        raise ValueError("invalid episodic GAE input")
    advantages = np.zeros_like(rewards)
    carry = np.zeros(8)
    for t in range(len(rewards) - 1, -1, -1):
        following = values[t + 1] if t + 1 < len(rewards) else np.zeros(8)
        carry = rewards[t] + gamma * following - values[t] + gamma * trace_decay * carry
        advantages[t] = carry
    return advantages, advantages + values


def update_private_actors(
    parent: NearBallResidualPolicy,
    rollouts: list[dict[str, Any]],
    *,
    epochs: int = 4,
    gamma: float = 0.99,
    trace_decay: float = 0.95,
    reward_shaping: str = "legacy",
    frozen_policy_hashes: Mapping[str, str] | None = None,
    behavior_anchor: RoleBehaviorAnchor | None = None,
    trainable_agent_ids: tuple[str, ...] | None = None,
) -> tuple[NearBallResidualPolicy, list[dict[str, Any]]]:
    # Optional training dependency: readers and the simulator use only NumPy.
    import torch

    torch.set_num_threads(1)
    if not rollouts or type(epochs) is not int or not 1 <= epochs <= 16:
        raise ValueError("PPO needs rollouts and bounded update epochs")
    scoped_learning = any("residual_learning_mask" in trace for trace in rollouts)
    if scoped_learning and not all("residual_learning_mask" in trace for trace in rollouts):
        raise ValueError("learning eligibility must be explicit for every rollout")
    if trainable_agent_ids is not None and (
        type(trainable_agent_ids) is not tuple
        or not trainable_agent_ids
        or any(type(agent) is not str for agent in trainable_agent_ids)
        or tuple(sorted(set(trainable_agent_ids))) != trainable_agent_ids
        or not set(trainable_agent_ids).issubset(parent.agent_ids)
    ):
        raise ValueError("trainable roles must be a nonempty canonical subset of the roster")
    if frozen_policy_hashes is not None and not isinstance(frozen_policy_hashes, Mapping):
        raise ValueError("frozen component bindings must be a mapping")
    frozen = dict(frozen_policy_hashes or {})
    if behavior_anchor is not None:
        if (
            not isinstance(behavior_anchor, RoleBehaviorAnchor)
            or behavior_anchor.policy.agent_ids != parent.agent_ids
            or behavior_anchor.policy.body_hash != parent.body_hash
            or behavior_anchor.policy.observation_contract != parent.observation_contract
            or "retention.anchor" in frozen
        ):
            raise ValueError("behavior anchor must match body/roster without binding aliases")
        frozen["retention.anchor"] = behavior_anchor.policy.policy_hash
    if (
        len(frozen) > 32
        or set(frozen).intersection(parent.agent_ids)
        or any(
            not isinstance(key, str)
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", key) is None
            or not isinstance(value, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
            for key, value in frozen.items()
        )
    ):
        raise ValueError("distinct content-bound frozen components required")
    weights = {k: v.copy() for k, v in parent.weights.items()}
    advantages, returns = [], []
    for trace in rollouts:
        _validate_on_policy(parent, trace)
        a, r = episodic_gae(
            physical_rewards(trace, parent.agent_ids, reward_shaping=reward_shaping, gamma=gamma),
            trace["residual_value"],
            gamma=gamma,
            trace_decay=trace_decay,
        )
        advantages.append(a)
        returns.append(r)
    observations = np.concatenate([t["residual_observations"] for t in rollouts])
    actions = np.concatenate([t["residual_latent"] for t in rollouts])
    logps = np.concatenate([t["residual_log_probability"] for t in rollouts])
    active = np.concatenate([t["residual_active"] for t in rollouts])
    learning = (
        np.concatenate([t["residual_learning_mask"] for t in rollouts]) if scoped_learning else None
    )
    scoped_exploration = any("residual_exploration_mask" in t for t in rollouts)
    exploration = np.concatenate(
        [
            t.get("residual_exploration_mask", np.ones_like(t["residual_active"], dtype=bool))
            for t in rollouts
        ]
    )
    advantage, target_value = np.concatenate(advantages), np.concatenate(returns)
    dataset_hash = str(hash_json({"rollouts": [trajectory_digest(t) for t in rollouts]}))
    context_hash = str(
        hash_json(
            {
                "body": parent.body_hash,
                "parent": parent.policy_hash,
                "roster": parent.agent_ids,
                "dataset": dataset_hash,
                "physical_reward_contract": PHYSICAL_REWARD_CONTRACT,
                **({"frozen_policy_hashes": frozen} if frozen else {}),
                **({"behavior_anchor": behavior_anchor.anchor_hash} if behavior_anchor else {}),
                **(
                    {"trainable_agent_ids": trainable_agent_ids}
                    if trainable_agent_ids is not None
                    else {}
                ),
                **(
                    {"reward_shaping": reward_shaping, "gamma": gamma}
                    if reward_shaping != "legacy"
                    else {}
                ),
            }
        )
    )
    rows = []
    for i, agent in enumerate(parent.agent_ids):
        mask = active[:, i].astype(bool)
        n = int(mask.sum())
        row: dict[str, Any] = {"agent_id": agent, "active_samples": n, "updated": False}
        if scoped_exploration:
            mask &= exploration[:, i]
            n = int(mask.sum())
            row["learning_samples"] = n
        if learning is not None:
            # Task eligibility is neither physical activation nor exploration.
            # Do not relabel sampled actions to exclude support/cover tasks.
            mask &= learning[:, i]
            n = int(mask.sum())
            row["learning_samples"] = n
            row["task_eligibility_applied"] = True
        if trainable_agent_ids is not None and agent not in trainable_agent_ids:
            # Keep the measured active mask truthful. Learning eligibility is
            # separate from which actor actually controlled the simulation.
            row["frozen_by_training_scope"] = True
            rows.append(row)
            continue
        if n < 32:
            rows.append(row)
            continue
        before_hashes = private_weight_hashes(weights, parent.agent_ids, parent.body_hash)
        before_hashes.update(frozen)
        lease = begin_update(
            before=before_hashes,
            focal=agent,
            generation=parent.generation + 1,
            dataset_hash=dataset_hash,
            context_hash=context_hash,
            maximum_steps=epochs,
        )
        optimizer_steps = 0
        parameters = {
            k: torch.nn.Parameter(torch.tensor(v[i], dtype=torch.float64))
            for k, v in weights.items()
        }
        optimizer = torch.optim.Adam(list(parameters.values()), lr=1e-4)
        x = torch.tensor(observations[mask, i], dtype=torch.float64)
        action = torch.tensor(actions[mask, i], dtype=torch.float64)
        old_logp = torch.tensor(logps[mask, i], dtype=torch.float64)
        adv = torch.tensor(advantage[mask, i], dtype=torch.float64)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        value_target = torch.tensor(target_value[mask, i], dtype=torch.float64)
        if behavior_anchor is not None:
            row["behavior_anchor"] = {
                "anchor_hash": behavior_anchor.anchor_hash,
                "samples": int(behavior_anchor.active[:, i].sum()),
                "coefficient": behavior_anchor.coefficient,
                "kl_before": float(behavior_anchor.kl_loss(parameters, i).detach()),
            }
        initial_mean = None
        for _ in range(epochs):
            hidden = torch.tanh(x @ parameters["w1"] + parameters["b1"])
            mean = hidden @ parameters["w2"] + parameters["b2"]
            if initial_mean is None:
                initial_mean = mean.detach().clone()
            logp = (
                -0.5 * ((action - mean) / parameters["log_std"].exp()).square()
                - parameters["log_std"]
                - 0.5 * math.log(2 * math.pi)
            ).sum(dim=1)
            ratio = torch.exp(logp - old_logp)
            approx_kl = ((ratio - 1) - (logp - old_logp)).mean()
            if float(approx_kl.detach()) > 0.015:
                break
            surrogate = torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv)
            value = hidden @ parameters["wv"] + parameters["bv"]
            loss = (
                -surrogate.mean()
                + 0.5 * (value - value_target).square().mean()
                + 0.1 * (mean - initial_mean).square().mean()
                - 0.0001 * parameters["log_std"].sum()
            )
            if behavior_anchor is not None:
                loss = loss + behavior_anchor.coefficient * behavior_anchor.kl_loss(parameters, i)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite PPO update rejected")
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 0.5)
            if not torch.isfinite(norm):
                raise ValueError("nonfinite PPO gradient rejected")
            optimizer.step()
            optimizer_steps += 1
            with torch.no_grad():
                parameters["log_std"].clamp_(-4, -0.2)
        for k, parameter in parameters.items():
            weights[k][i] = parameter.detach().numpy()
        if behavior_anchor is not None:
            row["behavior_anchor"]["kl_after"] = float(
                behavior_anchor.kl_loss(parameters, i).detach()
            )
        row["core_plasticity"] = finish_update(
            lease=lease,
            before=before_hashes,
            after={**private_weight_hashes(weights, parent.agent_ids, parent.body_hash), **frozen},
            steps=optimizer_steps,
        )
        delta = float(sum(np.square(weights[k][i] - parent.weights[k][i]).sum() for k in weights))
        row.update(
            updated=delta > 0,
            squared_parameter_delta=delta,
            shaping_return=float(
                sum(
                    physical_rewards(
                        t, parent.agent_ids, reward_shaping=reward_shaping, gamma=gamma
                    )[:, i].sum()
                    for t in rollouts
                )
            ),
        )
        rows.append(row)
    return NearBallResidualPolicy(
        parent.agent_ids,
        parent.body_hash,
        parent.generation + 1,
        parent.policy_hash,
        weights,
        parent.observation_contract,
    ), rows


def _trace_observation_dimension(trace: dict[str, Any], frames: int) -> int:
    if "residual_observation_contract_code" not in trace:
        return 56
    code = np.asarray(trace["residual_observation_contract_code"])
    if code.shape != (frames,) or code.dtype.kind not in "iu" or not np.all(code == 2):
        raise ValueError("invalid explicit residual observation contract")
    return 58


def _validate_on_policy(policy: NearBallResidualPolicy, trace: dict[str, Any]) -> None:
    obs = np.asarray(trace["residual_observations"])
    n = len(trace["time"])
    if _trace_observation_dimension(trace, n) != policy.observation_dim:
        raise ValueError("policy and rollout observation contracts differ")
    shapes = {
        "residual_observations": (n, 8, policy.observation_dim),
        "residual_latent": (n, 8, 12),
        "residual_value": (n, 8),
        "residual_log_probability": (n, 8),
        "residual_active": (n, 8),
        "residual_applied": (n, 8, 12),
    }
    for k, shape in shapes.items():
        if np.shape(trace[k]) != shape or not np.all(np.isfinite(trace[k])):
            raise ValueError("invalid on-policy rollout tensors")
    if np.asarray(trace["residual_active"]).dtype != np.bool_:
        raise ValueError("PPO activation mask must be boolean")
    if "residual_exploration_mask" in trace:
        mask = np.asarray(trace["residual_exploration_mask"])
        if mask.shape != (n, 8) or mask.dtype != np.bool_:
            raise ValueError("PPO exploration mask must be an explicit boolean frame/role array")
    if "residual_learning_mask" in trace:
        learning = np.asarray(trace["residual_learning_mask"])
        if learning.shape != (n, 8) or learning.dtype != np.bool_:
            raise ValueError("PPO learning mask must be an explicit boolean frame/role array")
    w = policy.weights
    hidden = np.tanh(np.einsum("tni,nij->tnj", obs, w["w1"]) + w["b1"])
    mean = np.einsum("tni,nij->tnj", hidden, w["w2"]) + w["b2"]
    if "residual_exploration_mask" in trace and not np.allclose(
        np.asarray(trace["residual_latent"])[~mask], mean[~mask], rtol=0, atol=1e-10
    ):
        raise ValueError("nonexploring roles must use the recorded parent's deterministic action")
    logp = (
        -0.5 * ((trace["residual_latent"] - mean) / np.exp(w["log_std"])) ** 2
        - w["log_std"]
        - 0.5 * math.log(2 * math.pi)
    ).sum(axis=2)
    values = (hidden * w["wv"]).sum(axis=2) + w["bv"]
    if not np.allclose(
        logp, trace["residual_log_probability"], rtol=0, atol=1e-10
    ) or not np.allclose(values, trace["residual_value"], rtol=0, atol=1e-10):
        raise ValueError("rollout was not sampled from this frozen parent policy")


def _collect(
    job: tuple[str, str, str, float, bool, int, float, bool, bool, bool],
    *,
    contact_control_profile: ContactControlProfile | None = None,
) -> str:
    (
        assets,
        destination,
        checkpoint,
        duration,
        blue,
        seed,
        offset,
        prospective,
        clearance,
        strict,
    ) = job
    run_probe(
        asset_root=Path(assets),
        output=Path(destination),
        active=True,
        four_vs_four=True,
        duration=duration,
        blue_kickoff=blue,
        near_ball_policy=NearBallResidualPolicy.load(Path(checkpoint)),
        near_ball_explore=True,
        near_ball_seed=seed,
        kickoff_offset_m=offset,
        anticipatory_contact=prospective,
        forward_receiver_lane=prospective,
        contact_policy=OwnedBallContactPolicy() if prospective else None,
        all_role_clearance=clearance,
        strict_receive_handoff=strict,
        contact_control_profile=contact_control_profile,
    )
    return str(Path(destination) / "probe.json")


def _bilateral_jobs(
    assets: Path,
    output: Path,
    generation: int,
    iteration: int,
    duration: float,
    offsets: tuple[float, ...],
    rollouts_per_side: int,
    prospective: bool,
    clearance: bool,
    strict: bool,
) -> list[tuple[str, str, str, float, bool, int, float, bool, bool, bool]]:
    """Bind diverse episodes to the same frozen generation, without overwriting artifacts."""
    jobs = []
    for round_index in range(rollouts_per_side):
        for blue in (False, True):
            side = "blue" if blue else "red"
            suffix = "" if rollouts_per_side == 1 else f"-{round_index:02d}"
            jobs.append(
                (
                    str(assets),
                    str(output / f"train-{iteration:03d}-{side}{suffix}"),
                    str(output / f"generation-{generation:03d}.npz"),
                    duration,
                    blue,
                    21500 + iteration * 2 * rollouts_per_side + round_index * 2 + int(blue),
                    offsets[(iteration * rollouts_per_side + round_index) % len(offsets)],
                    prospective,
                    clearance,
                    strict,
                )
            )
    return jobs


def _coverage_extra_jobs(
    assets: Path,
    output: Path,
    generation: int,
    iteration: int,
    duration: float,
    offsets: tuple[float, ...],
    round_index: int,
    prospective: bool,
    clearance: bool,
    strict: bool,
) -> list[tuple[str, str, str, float, bool, int, float, bool, bool, bool]]:
    """Fresh bilateral rollouts from the exact same frozen parent generation."""
    if not offsets or type(round_index) is not int or not 0 <= round_index < 8:
        raise ValueError("bounded explicit extra coverage round required")
    return [
        (
            str(assets),
            str(
                output
                / f"train-{iteration:03d}-coverage-{round_index:02d}-{'blue' if blue else 'red'}"
            ),
            str(output / f"generation-{generation:03d}.npz"),
            duration,
            blue,
            35000 + iteration * 16 + round_index * 2 + int(blue),
            offsets[(iteration * 8 + round_index) % len(offsets)],
            prospective,
            clearance,
            strict,
        )
        for blue in (False, True)
    ]


def train(
    *,
    assets: Path,
    output: Path,
    iterations: int,
    duration: float,
    prospective_curriculum: bool = False,
    workers: int = 1,
    long_credit: bool = False,
    all_role_clearance: bool = False,
    diverse_ball_positions: bool = False,
    initial_checkpoint: Path | None = None,
    role_curriculum: bool = False,
    strict_receive_handoff: bool = False,
    role_batch_rounds: int = 1,
    rollouts_per_side: int = 1,
    reward_shaping: str = "legacy",
    contact_control_profile: ContactControlProfile | None = None,
    optimizer_epochs: int = 4,
    trainable_agent_ids: tuple[str, ...] | None = None,
    minimum_actor_samples: int = 0,
    maximum_coverage_pairs: int = 0,
    receiving_learning_only: bool = False,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if (
        type(iterations) is not int
        or type(optimizer_epochs) is not int
        or not 1 <= optimizer_epochs <= 16
        or not 1 <= iterations <= 100
        or not 5 <= duration <= 25
        or type(rollouts_per_side) is not int
        or not 1 <= rollouts_per_side <= 8
        or type(workers) is not int
        or not 1 <= workers <= (8 if role_curriculum or rollouts_per_side > 1 else 2)
        or type(prospective_curriculum) is not bool
        or type(long_credit) is not bool
        or type(all_role_clearance) is not bool
        or type(diverse_ball_positions) is not bool
        or type(role_curriculum) is not bool
        or type(strict_receive_handoff) is not bool
        or type(receiving_learning_only) is not bool
        or (receiving_learning_only and trainable_agent_ids is None)
        or type(role_batch_rounds) is not int
        or not 1 <= role_batch_rounds <= 5
        or (role_batch_rounds != 1 and not role_curriculum)
        or (rollouts_per_side != 1 and role_curriculum)
        or reward_shaping not in REWARD_SHAPING_MODES
        or reward_shaping == "in_play_motor_task_v1"
        or (reward_shaping == "directed_pass_credit_v1" and not strict_receive_handoff)
        or (strict_receive_handoff and not (prospective_curriculum and all_role_clearance))
        or type(minimum_actor_samples) is not int
        or type(maximum_coverage_pairs) is not int
        or not 0 <= maximum_coverage_pairs <= 8
        or (
            (minimum_actor_samples, maximum_coverage_pairs) != (0, 0)
            and (
                not 32 <= minimum_actor_samples <= 4096
                or maximum_coverage_pairs == 0
                or role_curriculum
                or trainable_agent_ids is None
                or not strict_receive_handoff
            )
        )
        or (role_curriculum and not (prospective_curriculum and all_role_clearance))
        or (
            contact_control_profile is not None
            and not isinstance(contact_control_profile, ContactControlProfile)
        )
    ):
        raise ValueError("bounded online training budget required")
    fixture = build_four_vs_four_fixture(assets)
    ids = tuple(sorted(c.agent_id for c in fixture.cells))
    if trainable_agent_ids is not None and (
        type(trainable_agent_ids) is not tuple
        or not trainable_agent_ids
        or any(type(agent) is not str for agent in trainable_agent_ids)
        or tuple(sorted(set(trainable_agent_ids))) != trainable_agent_ids
        or not set(trainable_agent_ids).issubset(ids)
    ):
        raise ValueError("trainable actors must be a canonical nonempty roster subset")
    body = fixture.cells[0].growth_scope.body_hash
    policy = (
        NearBallResidualPolicy.initialize(ids, body)
        if initial_checkpoint is None
        else NearBallResidualPolicy.load(initial_checkpoint)
    )
    if (
        policy.agent_ids != ids
        or policy.body_hash != body
        or policy.generation + iterations > 1000000
    ):
        raise ValueError("initial candidate does not match the training body/roster/budget")
    initial_policy = policy
    offsets: tuple[float, ...] = (
        (-0.16, -0.12, -0.04, 0.0, 0.04, 0.12, 0.16) if diverse_ball_positions else (-0.04, 0.04)
    )
    if role_curriculum:
        offsets = TRAIN_OFFSETS
    output.mkdir(parents=True)
    policy.save(output / f"generation-{policy.generation:03d}.npz")
    manifest: dict[str, Any] = {
        "schema": "rosclaw_soccer.private_near_ball_ppo.v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
        "iterations": [],
        "evaluation": [],
        "training_source_hash": hash_bytes(Path(__file__).read_bytes()),
        "prospective_curriculum": prospective_curriculum,
        "workers": workers,
        "optimizer_epochs": optimizer_epochs,
        "trainable_agent_ids": trainable_agent_ids,
        "initial_generation": policy.generation,
        "initial_policy_hash": policy.policy_hash,
        "baseline_label": "parent" if initial_checkpoint is not None else "zero",
        "initial_checkpoint_hash": None
        if initial_checkpoint is None
        else hash_bytes(initial_checkpoint.read_bytes()),
        "all_role_clearance": all_role_clearance,
        "role_curriculum": role_curriculum,
        "role_batch_rounds": role_batch_rounds,
        "rollouts_per_side": rollouts_per_side,
        "contact_control_profile": None
        if contact_control_profile is None
        else asdict(contact_control_profile),
        "reward_shaping": reward_shaping,
        "physical_reward_contract": PHYSICAL_REWARD_CONTRACT,
        "strict_receive_handoff": strict_receive_handoff,
        "evaluation_courses": [
            asdict(c) for c in examination_courses(strict_handoff=strict_receive_handoff)
        ]
        if role_curriculum
        else None,
        "training_offsets_m": offsets,
        "credit": {
            "gamma": 0.997 if long_credit else 0.99,
            "trace_decay": 0.997 if long_credit else 0.95,
            "control_dt_sec": 0.02,
        },
        "note": "Residual PPO, frozen locomotion; not an end-to-end torque policy.",
    }
    if minimum_actor_samples:
        from rosclaw_soccer.training import role_sample_coverage as coverage_module

        manifest["adaptive_coverage"] = {
            "minimum_actor_samples": minimum_actor_samples,
            "maximum_coverage_pairs": maximum_coverage_pairs,
            "coverage_source_hash": hash_bytes(Path(coverage_module.__file__).read_bytes()),
            "semantics": "on-policy physical/exploration/learning mask counts; not task success",
            "iterations": [],
        }
    if receiving_learning_only:
        from rosclaw_soccer.training import receiving_learning_scope as receiving_scope_module

        manifest["receiving_learning_scope"] = {
            "source_hash": hash_bytes(Path(receiving_scope_module.__file__).read_bytes()),
            "intent": ["receive", "press", "intercept"],
            "semantics": "recorded current intent gates PPO gradients; full-episode returns",
        }
    _persist_training_manifest(output, manifest)
    bilateral_collect = (
        _collect
        if contact_control_profile is None
        else partial(_collect, contact_control_profile=contact_control_profile)
    )
    for iteration in range(iterations):
        traces, proofs = [], []
        jobs = _bilateral_jobs(
            assets,
            output,
            policy.generation,
            iteration,
            duration,
            offsets,
            rollouts_per_side,
            prospective_curriculum,
            all_role_clearance,
            strict_receive_handoff,
        )
        if role_curriculum:
            role_jobs = [
                RoleRolloutJob(
                    assets,
                    output / f"train-{iteration:03d}-{course.key}",
                    output / f"generation-{policy.generation:03d}.npz",
                    duration,
                    course,
                    22100 + iteration * role_batch_rounds * 8 + index,
                    True,
                    strict_receive_handoff,
                    contact_control_profile,
                )
                for index, course in enumerate(training_batch(iteration, rounds=role_batch_rounds))
            ]
            if workers == 1:
                collected = [collect_role_course(job) for job in role_jobs]
            else:
                with ProcessPoolExecutor(
                    max_workers=workers, mp_context=multiprocessing.get_context("spawn")
                ) as pool:
                    collected = list(pool.map(collect_role_course, role_jobs))
        elif workers == 1:
            collected = [bilateral_collect(job) for job in jobs]
        else:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                collected = list(pool.map(bilateral_collect, jobs))
        for source in collected:
            destination = Path(source).parent
            report = validate_probe(Path(source))
            _verify_contact_profile(report, contact_control_profile)
            if not report["exact_replay"]:
                raise ValueError("nonreproducible rollout rejected")
            with np.load(destination / "primary.npz", allow_pickle=False) as archive:
                trace = {k: archive[k] for k in archive.files}
            if receiving_learning_only:
                from rosclaw_soccer.training.receiving_learning_scope import (
                    receiving_learning_scope,
                )

                trace = receiving_learning_scope(trace, agent_ids=ids)
            traces.append(trace)
            proofs.append(report["report_hash"])
        if minimum_actor_samples:
            from rosclaw_soccer.training.role_sample_coverage import role_sample_coverage

            if trainable_agent_ids is None:
                raise ValueError("explicit training roles required for coverage")
            for trace in traces:
                _validate_on_policy(policy, trace)
            counts = role_sample_coverage(traces, agent_ids=ids, required=trainable_agent_ids)
            coverage: dict[str, Any] = {
                "initial_counts": counts,
                "extra_rounds": [],
                "complete": False,
            }
            manifest["adaptive_coverage"]["iterations"].append(coverage)
            _persist_training_manifest(output, manifest)
            for extra in range(maximum_coverage_pairs):
                if all(count >= minimum_actor_samples for count in counts.values()):
                    break
                extra_jobs = _coverage_extra_jobs(
                    assets,
                    output,
                    policy.generation,
                    iteration,
                    duration,
                    offsets,
                    extra,
                    prospective_curriculum,
                    all_role_clearance,
                    strict_receive_handoff,
                )
                if workers == 1:
                    more = [bilateral_collect(job) for job in extra_jobs]
                else:
                    with ProcessPoolExecutor(
                        max_workers=min(workers, 2), mp_context=multiprocessing.get_context("spawn")
                    ) as pool:
                        more = list(pool.map(bilateral_collect, extra_jobs))
                extra_hashes = []
                for source in more:
                    report = validate_probe(Path(source))
                    _verify_contact_profile(report, contact_control_profile)
                    if not report["exact_replay"]:
                        raise ValueError("nonreproducible coverage rollout rejected")
                    with np.load(
                        Path(source).parent / "primary.npz", allow_pickle=False
                    ) as archive:
                        trace = {k: archive[k] for k in archive.files}
                    if receiving_learning_only:
                        from rosclaw_soccer.training.receiving_learning_scope import (
                            receiving_learning_scope,
                        )

                        trace = receiving_learning_scope(trace, agent_ids=ids)
                    _validate_on_policy(policy, trace)
                    traces.append(trace)
                    proofs.append(report["report_hash"])
                    extra_hashes.append(report["report_hash"])
                counts = role_sample_coverage(traces, agent_ids=ids, required=trainable_agent_ids)
                coverage["extra_rounds"].append(
                    {"round": extra, "counts": counts, "reports": extra_hashes}
                )
                _persist_training_manifest(output, manifest)
            coverage["complete"] = all(count >= minimum_actor_samples for count in counts.values())
            coverage["final_counts"] = counts
            _persist_training_manifest(output, manifest)
            if not coverage["complete"]:
                raise ValueError("actor coverage exhausted: no update for this generation")
        policy, rows = update_private_actors(
            policy,
            traces,
            epochs=optimizer_epochs,
            gamma=manifest["credit"]["gamma"],
            trace_decay=manifest["credit"]["trace_decay"],
            reward_shaping=reward_shaping,
            trainable_agent_ids=trainable_agent_ids,
        )
        policy.save(output / f"generation-{policy.generation:03d}.npz")
        manifest["iterations"].append(
            {
                "generation": policy.generation,
                "policy_hash": policy.policy_hash,
                "rollout_report_hashes": proofs,
                "players": rows,
            }
        )
        _persist_training_manifest(output, manifest)
        print(json.dumps(manifest["iterations"][-1]), flush=True)
    for label, candidate in ((manifest["baseline_label"], initial_policy), ("candidate", policy)):
        if role_curriculum:
            exam_jobs = [
                RoleRolloutJob(
                    assets,
                    output / f"eval-{label}-{course.key}",
                    output / f"generation-{candidate.generation:03d}.npz",
                    duration,
                    course,
                    0,
                    False,
                    strict_receive_handoff,
                    contact_control_profile,
                )
                for course in examination_courses(strict_handoff=strict_receive_handoff)
            ]
            if workers == 1:
                sources = [collect_role_course(job) for job in exam_jobs]
            else:
                with ProcessPoolExecutor(
                    max_workers=workers, mp_context=multiprocessing.get_context("spawn")
                ) as pool:
                    sources = list(pool.map(collect_role_course, exam_jobs))
            for job, source in zip(exam_jobs, sources, strict=True):
                report = validate_probe(Path(source))
                _verify_contact_profile(report, contact_control_profile)
                manifest["evaluation"].append(
                    {
                        "label": label,
                        **asdict(job.course),
                        "report_hash": report["report_hash"],
                        "safe": report["results"][0]["safe"],
                        "assessment": report["assessment"],
                        "passes": report["causal_pass_feedback"],
                    }
                )
                _persist_training_manifest(output, manifest)
            continue
        for blue in (False, True):
            for offset in (-0.08, 0.08):
                destination = output / f"eval-{label}-{'blue' if blue else 'red'}-{offset:+.2f}"
                report = run_probe(
                    asset_root=assets,
                    output=destination,
                    active=True,
                    four_vs_four=True,
                    duration=duration,
                    blue_kickoff=blue,
                    near_ball_policy=candidate,
                    kickoff_offset_m=offset,
                    anticipatory_contact=prospective_curriculum,
                    forward_receiver_lane=prospective_curriculum,
                    contact_policy=OwnedBallContactPolicy() if prospective_curriculum else None,
                    all_role_clearance=all_role_clearance,
                    strict_receive_handoff=strict_receive_handoff,
                    contact_control_profile=contact_control_profile,
                )
                checked = validate_probe(destination / "probe.json")
                _verify_contact_profile(checked, contact_control_profile)
                manifest["evaluation"].append(
                    {
                        "label": label,
                        "blue": blue,
                        "offset": offset,
                        "report_hash": report["report_hash"],
                        "safe": report["results"][0]["safe"],
                        "assessment": report["assessment"],
                        "passes": report["causal_pass_feedback"],
                    }
                )
                _persist_training_manifest(output, manifest)
    manifest["manifest_hash"] = hash_json(manifest)
    _persist_training_manifest(output, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--duration", type=float, default=12)
    parser.add_argument("--prospective-curriculum", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--long-credit", action="store_true")
    parser.add_argument("--all-role-clearance", action="store_true")
    parser.add_argument("--diverse-ball-positions", action="store_true")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--role-curriculum", action="store_true")
    parser.add_argument("--strict-receive-handoff", action="store_true")
    parser.add_argument("--role-batch-rounds", type=int, default=1)
    parser.add_argument("--rollouts-per-side", type=int, default=1)
    parser.add_argument("--contact-control-profile", action="store_true")
    parser.add_argument("--strike-residual", action="store_true")
    parser.add_argument("--strike-stance-lateral", type=float)
    parser.add_argument("--optimizer-epochs", type=int, default=4)
    parser.add_argument("--minimum-actor-samples", type=int, default=0)
    parser.add_argument("--maximum-coverage-pairs", type=int, default=0)
    parser.add_argument("--receiving-learning-only", action="store_true")
    parser.add_argument(
        "--trainable-agent",
        action="append",
        help="Explicit private actor to update; repeat for each actor. Others remain frozen.",
    )
    parser.add_argument(
        "--reward-shaping",
        choices=tuple(mode for mode in REWARD_SHAPING_MODES if mode != "in_play_motor_task_v1"),
        default="legacy",
    )
    args = parser.parse_args()
    if args.strike_residual and not args.contact_control_profile:
        parser.error("--strike-residual requires --contact-control-profile")
    if args.strike_stance_lateral is not None and not args.contact_control_profile:
        parser.error("--strike-stance-lateral requires --contact-control-profile")
    train(
        assets=args.asset_root,
        output=args.output,
        iterations=args.iterations,
        duration=args.duration,
        prospective_curriculum=args.prospective_curriculum,
        workers=args.workers,
        long_credit=args.long_credit,
        all_role_clearance=args.all_role_clearance,
        diverse_ball_positions=args.diverse_ball_positions,
        initial_checkpoint=args.initial_checkpoint,
        role_curriculum=args.role_curriculum,
        strict_receive_handoff=args.strict_receive_handoff,
        role_batch_rounds=args.role_batch_rounds,
        rollouts_per_side=args.rollouts_per_side,
        receiving_learning_only=args.receiving_learning_only,
        reward_shaping=args.reward_shaping,
        contact_control_profile=ContactControlProfile(
            strike_residual_enabled=args.strike_residual,
            strike_stance_lateral_m=args.strike_stance_lateral,
        )
        if args.contact_control_profile
        else None,
        optimizer_epochs=args.optimizer_epochs,
        minimum_actor_samples=args.minimum_actor_samples,
        maximum_coverage_pairs=args.maximum_coverage_pairs,
        trainable_agent_ids=None
        if args.trainable_agent is None
        else tuple(sorted(args.trainable_agent)),
    )


if __name__ == "__main__":
    main()
