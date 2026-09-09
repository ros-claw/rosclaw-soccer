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
import re
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.growth.owned_ball_contact import OwnedBallContactPolicy
from rosclaw_soccer.growth.pass_failure_feedback import diagnose_passes
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
from rosclaw_soccer.training.near_ball_curriculum import (
    TRAIN_OFFSETS,
    RoleRolloutJob,
    collect_role_course,
    examination_courses,
    training_batch,
)
from rosclaw_soccer.training.near_ball_plasticity import (
    begin_update,
    finish_update,
    private_weight_hashes,
)


def physical_rewards(
    trace: dict[str, Any],
    ids: tuple[str, ...],
    *,
    reward_shaping: str = "legacy",
    gamma: float = 0.99,
) -> np.ndarray:
    """Foot approach + directed contact, minus tilt, impacts and residual effort.

    No reward for a planner merely declaring PASS. Ball motion is rewarded only
    on this player's measured foot-contact frames. Dense contact is capped by
    the control period, so resting on the ball cannot earn an event bonus.
    """
    obs = np.asarray(trace["residual_observations"], dtype=float)
    if reward_shaping not in REWARD_SHAPING_MODES:
        raise ValueError("unknown reward shaping contract")
    count = len(trace["time"])
    if obs.shape != (count, 8, 56) or not np.all(np.isfinite(obs)):
        raise ValueError("PPO observations are not finite physical samples")
    rewards = np.zeros((count, 8))
    ball = np.asarray(trace["ball_pose"])[:, :3]
    contact = np.asarray(trace["ball_contact_agent_code"])
    foot = np.isin(trace["ball_contact_effector_code"], [1, 2])
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
        target = np.asarray(trace[key + "_target_position"])[:, :2]
        direction = target - ball[:, :2]
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
        toward = (np.asarray(trace["ball_velocity"])[:, :2] * direction).sum(axis=1)
        touching = (contact == i + 1) & foot
        rewards[:, i] += touching * 0.04 * np.clip(toward, -2, 2)
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
    if launch_relative:
        contract = np.asarray(trace["pass_feedback_launch_relative"])
        if contract.shape != time.shape or contract.dtype != np.bool_ or not np.all(contract):
            raise ValueError("physical reward handoff contract is invalid")
    for outcome in diagnose_passes(trace, ids, launch_relative=launch_relative):
        if not outcome["physical_receive_confirmed"]:
            continue
        sender, receiver = ids.index(outcome["sender"]), ids.index(outcome["receiver"])
        first = int(np.searchsorted(time, outcome["foot_contact_time_sec"]))
        received = np.flatnonzero(
            (contact == receiver + 1)
            & foot
            & (np.asarray(trace["ball_contact_force_n"]) > 0)
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
    if reward_shaping in {"terminal_potential_v1", "contact_safety_v1"}:
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
    if reward_shaping == "contact_safety_v1":
        margins = np.stack(
            [np.asarray(trace[a.replace(".", "_") + "_joint_safety_margin_rad"]) for a in ids],
            axis=1,
        )
        if margins.shape != (count, 8, 29):
            raise ValueError("joint margin trace does not match reward observations")
        rewards += joint_safety_penalty(margins)
    return rewards


def episodic_gae(
    rewards: np.ndarray, values: np.ndarray, *, gamma: float = 0.99, trace_decay: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
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
) -> tuple[NearBallResidualPolicy, list[dict[str, Any]]]:
    # Optional training dependency: readers and the simulator use only NumPy.
    import torch

    torch.set_num_threads(1)
    if not rollouts or type(epochs) is not int or not 1 <= epochs <= 16:
        raise ValueError("PPO needs rollouts and bounded update epochs")
    if frozen_policy_hashes is not None and not isinstance(frozen_policy_hashes, Mapping):
        raise ValueError("frozen component bindings must be a mapping")
    frozen = dict(frozen_policy_hashes or {})
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
    advantage, target_value = np.concatenate(advantages), np.concatenate(returns)
    dataset_hash = str(hash_json({"rollouts": [trajectory_digest(t) for t in rollouts]}))
    context_hash = str(
        hash_json(
            {
                "body": parent.body_hash,
                "parent": parent.policy_hash,
                "roster": parent.agent_ids,
                "dataset": dataset_hash,
                **({"frozen_policy_hashes": frozen} if frozen else {}),
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
        parent.agent_ids, parent.body_hash, parent.generation + 1, parent.policy_hash, weights
    ), rows


def _validate_on_policy(policy: NearBallResidualPolicy, trace: dict[str, Any]) -> None:
    obs = np.asarray(trace["residual_observations"])
    n = len(trace["time"])
    shapes = {
        "residual_observations": (n, 8, 56),
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
    w = policy.weights
    hidden = np.tanh(np.einsum("tni,nij->tnj", obs, w["w1"]) + w["b1"])
    mean = np.einsum("tni,nij->tnj", hidden, w["w2"]) + w["b2"]
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


def _collect(job: tuple[str, str, str, float, bool, int, float, bool, bool]) -> str:
    assets, destination, checkpoint, duration, blue, seed, offset, prospective, clearance = job
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
    )
    return str(Path(destination) / "probe.json")


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
    reward_shaping: str = "legacy",
    contact_control_profile: ContactControlProfile | None = None,
    optimizer_epochs: int = 4,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if (
        type(iterations) is not int
        or type(optimizer_epochs) is not int
        or not 1 <= optimizer_epochs <= 16
        or not 1 <= iterations <= 100
        or not 5 <= duration <= 25
        or type(workers) is not int
        or not 1 <= workers <= (8 if role_curriculum else 2)
        or type(prospective_curriculum) is not bool
        or type(long_credit) is not bool
        or type(all_role_clearance) is not bool
        or type(diverse_ball_positions) is not bool
        or type(role_curriculum) is not bool
        or type(strict_receive_handoff) is not bool
        or type(role_batch_rounds) is not int
        or not 1 <= role_batch_rounds <= 5
        or (role_batch_rounds != 1 and not role_curriculum)
        or reward_shaping not in REWARD_SHAPING_MODES
        or (strict_receive_handoff and not role_curriculum)
        or (role_curriculum and not (prospective_curriculum and all_role_clearance))
        or (
            contact_control_profile is not None
            and (
                not isinstance(contact_control_profile, ContactControlProfile)
                or not role_curriculum
            )
        )
    ):
        raise ValueError("bounded online training budget required")
    fixture = build_four_vs_four_fixture(assets)
    ids = tuple(sorted(c.agent_id for c in fixture.cells))
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
        "initial_generation": policy.generation,
        "initial_policy_hash": policy.policy_hash,
        "baseline_label": "parent" if initial_checkpoint is not None else "zero",
        "initial_checkpoint_hash": None
        if initial_checkpoint is None
        else hash_bytes(initial_checkpoint.read_bytes()),
        "all_role_clearance": all_role_clearance,
        "role_curriculum": role_curriculum,
        "role_batch_rounds": role_batch_rounds,
        "contact_control_profile": None
        if contact_control_profile is None
        else asdict(contact_control_profile),
        "reward_shaping": reward_shaping,
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
    for iteration in range(iterations):
        traces, proofs = [], []
        jobs = [
            (
                str(assets),
                str(output / f"train-{iteration:03d}-{'blue' if blue else 'red'}"),
                str(output / f"generation-{policy.generation:03d}.npz"),
                duration,
                blue,
                21500 + iteration * 2 + int(blue),
                offsets[iteration % len(offsets)],
                prospective_curriculum,
                all_role_clearance,
            )
            for blue in (False, True)
        ]
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
            collected = [_collect(job) for job in jobs]
        else:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                collected = list(pool.map(_collect, jobs))
        for source in collected:
            destination = Path(source).parent
            report = validate_probe(Path(source))
            if not report["exact_replay"]:
                raise ValueError("nonreproducible rollout rejected")
            with np.load(destination / "primary.npz", allow_pickle=False) as archive:
                traces.append({k: archive[k] for k in archive.files})
            proofs.append(report["report_hash"])
        policy, rows = update_private_actors(
            policy,
            traces,
            epochs=optimizer_epochs,
            gamma=manifest["credit"]["gamma"],
            trace_decay=manifest["credit"]["trace_decay"],
            reward_shaping=reward_shaping,
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
        (output / "training.json").write_text(json.dumps(manifest, indent=2) + "\n")
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
            (output / "training.json").write_text(json.dumps(manifest, indent=2) + "\n")
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
                )
                validate_probe(destination / "probe.json")
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
    manifest["manifest_hash"] = hash_json(manifest)
    (output / "training.json").write_text(json.dumps(manifest, indent=2) + "\n")
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
    parser.add_argument("--contact-control-profile", action="store_true")
    parser.add_argument("--strike-residual", action="store_true")
    parser.add_argument("--strike-stance-lateral", type=float)
    parser.add_argument("--optimizer-epochs", type=int, default=4)
    parser.add_argument("--reward-shaping", choices=REWARD_SHAPING_MODES, default="legacy")
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
        reward_shaping=args.reward_shaping,
        contact_control_profile=ContactControlProfile(
            strike_residual_enabled=args.strike_residual,
            strike_stance_lateral_m=args.strike_stance_lateral,
        )
        if args.contact_control_profile
        else None,
        optimizer_epochs=args.optimizer_epochs,
    )


if __name__ == "__main__":
    main()
