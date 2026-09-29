"""SIM_ONLY online 48D actor-critic with phase-aligned receiving returns."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_course_map_v190 import COURSES
from rsi_r1_kinematic_advantage_v201 import KinematicActor
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy, _run_task, _tasks
from rsi_r1_temporal_actor_critic_v193 import clean, score
from rsi_r1_temporal_self_imitation_v196 import FRESH_COURSES as CONSUMED_V202_COURSES
from rsi_r1_temporal_zero_v192 import TRAIN_COURSES

from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import KinematicMotorWeights
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import TemporalMotorWeights
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

SCHEMA = "rosclaw_soccer.rsi.r1_kinematic_online_ppo_v204.result.v1"
UPDATES = 2
EXPLORATION_STD = 0.12
SEED = 204001


def _terminal_quality(row: dict[str, Any]) -> float:
    if not row["safe"] or row["fault_agents"]:
        return -8.0
    if row["first_foot_frame"] is None:
        return -5.0
    if row["own_nonfoot_frames"]:
        return -4.0 - 0.2 * min(len(row["own_nonfoot_frames"]), 10)
    distance = min(float(row["tail_maximum_foot_distance_m"]), 2.0)
    speed = min(float(row["tail_maximum_ball_speed_mps"]), 3.0)
    return 2.0 + 7.0 * float(row["controlled_reception"]) - 1.5 * distance - speed


def _returns(row: dict[str, Any]) -> list[float]:
    frames = row["observed_frames"]
    shaped = np.asarray(row["shaped_reward"], dtype=np.float64)
    terminal = _terminal_quality(row)
    if len(shaped) != 100:
        return [terminal] * len(frames)
    tail = np.empty(101, dtype=np.float64)
    tail[100] = terminal
    for index in range(99, -1, -1):
        tail[index] = float(shaped[index]) + 0.99 * tail[index + 1]
    return [float(tail[max(0, min(frame - 20, 100))]) for frame in frames]


def _update(
    actor: KinematicActor,
    critic: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    episodes: list[dict[str, Any]],
    reference: KinematicActor,
) -> dict[str, float]:
    states = torch.tensor(
        [state for row in episodes for state in row["observed_features"]], dtype=torch.float32
    )
    actions = torch.tensor(
        [action for row in episodes for action in row["sampled_logits"]], dtype=torch.float32
    )
    returns = torch.tensor(
        [value for row in episodes for value in _returns(row)], dtype=torch.float32
    )
    if states.ndim != 2 or states.shape[1] != 48 or actions.shape != (len(states), 12):
        raise ValueError("complete aligned 48D online episodes required")
    with torch.no_grad():
        old_mean = actor(states)
        old_logp = torch.distributions.Normal(old_mean, EXPLORATION_STD).log_prob(actions).sum(-1)
        advantages = returns - critic(states).squeeze(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        reference_mean = reference(states)
    actor_loss = torch.tensor(0.0)
    critic_loss = torch.tensor(0.0)
    for _ in range(4):
        for indices in torch.randperm(len(states)).split(512):
            mean = actor(states[indices])
            new_logp = (
                torch.distributions.Normal(mean, EXPLORATION_STD).log_prob(actions[indices]).sum(-1)
            )
            ratio = (new_logp - old_logp[indices]).exp()
            ppo = -torch.minimum(
                ratio * advantages[indices],
                ratio.clamp(0.8, 1.2) * advantages[indices],
            ).mean()
            trust = (mean - reference_mean[indices]).square().mean()
            actor_loss = ppo + 0.2 * trust
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_optimizer.step()
            critic_loss = torch.nn.functional.mse_loss(
                critic(states[indices]).squeeze(-1), returns[indices]
            )
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            critic_optimizer.step()
    return {
        "actor_loss": float(actor_loss.detach()),
        "critic_loss": float(critic_loss.detach()),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
    }


def _save(path: Path, weights: KinematicMotorWeights) -> str:
    np.savez_compressed(
        path,
        input_matrix=np.asarray(weights.input_matrix),
        input_bias=np.asarray(weights.input_bias),
        output_matrix=np.asarray(weights.output_matrix),
        output_bias=np.asarray(weights.output_bias),
    )
    return str(hash_bytes(path.read_bytes()))


def train(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    zero_report_path: Path,
    v202_report_path: Path,
    v203_report_path: Path,
    base_policy_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY online kinematic evidence required")
    bank_hash = preflight_receiving_courses(TRAIN_COURSES)
    consumed_hash = preflight_receiving_courses(CONSUMED_V202_COURSES)
    parent, right_parent, refine, lateral, zero, v202, v203 = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            zero_report_path,
            v202_report_path,
            v203_report_path,
        )
    )
    if (
        v202["status"] != "REJECTED_PROTECTED_KINEMATIC_FRESH_GATE"
        or v203["status"] != "REJECTED_RANKED_UPDATE_GATE"
        or v203["v202_report_hash"] != v202["report_hash"]
        or v202["development_bank_hash"] != bank_hash
        or v202["fresh_bank_hash"] != consumed_hash
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
        or v202["development_score"][:3] != [1, 4, 10]
    ):
        raise ValueError("sealed rejected online curriculum lineage required")
    base = _load_policy(base_policy_path, v202["learned_policy_hash"])
    protected = tuple(
        tuple(float(value) for value in row["observed_features"][0]) for row in zero["rows"][-2:]
    )
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_kinematic_online_ppo_v204.py",
            "scripts/rsi_r1_kinematic_advantage_v201.py",
            "scripts/rsi_r1_protected_kinematic_fresh_v202.py",
            "scripts/rsi_r1_temporal_actor_critic_v193.py",
            "src/rosclaw_soccer/rsi/receiving_kinematic_temporal_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    actor = KinematicActor(base)
    reference = KinematicActor(base)
    critic = torch.nn.Sequential(torch.nn.Linear(48, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0002)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0005)
    output.mkdir(parents=True)
    history = []
    sampled_courses = (
        *COURSES,
        *CONSUMED_V202_COURSES,
        *(course for course in COURSES if course.speed_mps >= 1.28),
    )
    assert len(sampled_courses) == 32
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        baseline = list(
            pool.map(
                _run_task,
                _tasks(
                    asset_root,
                    policy_path,
                    TRAIN_COURSES,
                    coordination,
                    left,
                    right,
                    slope,
                    base,
                    protected,
                ),
            )
        )
        if tuple(score(baseline)[:3]) != tuple(v202["development_score"][:3]):
            raise ValueError("online start policy does not reproduce sealed development baseline")
        for update in range(UPDATES):
            seeds = tuple(int(value) for value in rng.integers(0, 2**31, size=len(sampled_courses)))
            tasks = [
                (
                    asset_root,
                    policy_path,
                    course,
                    coordination,
                    left,
                    right,
                    slope,
                    TemporalMotorWeights(),
                    EXPLORATION_STD,
                    seed,
                    0.0,
                    actor.weights(),
                    protected,
                )
                for course, seed in zip(sampled_courses, seeds, strict=True)
            ]
            sampled = list(pool.map(_run_task, tasks))
            optimization = _update(
                actor, critic, actor_optimizer, critic_optimizer, sampled, reference
            )
            weights = actor.weights()
            checkpoint_hash = _save(output / f"update-{update + 1}.npz", weights)
            deterministic = list(
                pool.map(
                    _run_task,
                    _tasks(
                        asset_root,
                        policy_path,
                        TRAIN_COURSES,
                        coordination,
                        left,
                        right,
                        slope,
                        weights,
                        protected,
                    ),
                )
            )
            retention_equal = all(
                row["protected_episode"] is True
                and row["physical_trace_hash"] == old["physical_trace_hash"]
                for row, old in zip(deterministic[-2:], zero["rows"][-2:], strict=True)
            )
            if not retention_equal:
                raise ValueError("online PPO violated hard old-skill protection")
            item = {
                "update": update + 1,
                "episodes": len(sampled),
                "sampled_clean": sum(clean(row) for row in sampled),
                "sampled_controlled": sum(
                    clean(row) and row["controlled_reception"] for row in sampled
                ),
                "sampled_high_speed_controlled": sum(
                    clean(row)
                    and row["controlled_reception"]
                    and row["course"]["speed_mps"] >= 1.28
                    for row in sampled
                ),
                "deterministic_score": score(deterministic),
                "checkpoint_hash": checkpoint_hash,
                "optimization": optimization,
                "retention_physical_equal": True,
            }
            history.append(item)
            (output / "progress.json").write_text(json.dumps(history, indent=2) + "\n")
            print(json.dumps(item), flush=True)
    best = max(history, key=lambda item: tuple(item["deterministic_score"]))
    qualified = (
        tuple(best["deterministic_score"]) > tuple(v202["development_score"])
        and best["deterministic_score"][1] >= 5
    )
    report = {
        "schema": SCHEMA,
        "v202_report_hash": v202["report_hash"],
        "v203_report_hash": v203["report_hash"],
        "source_hashes": sources,
        "partition": "ONLINE_PPO_CONSUMED_EIGHT_G1_CURRICULUM_NOT_FRESH",
        "course_bank_hash": bank_hash,
        "consumed_v202_course_hash": consumed_hash,
        "seed": SEED,
        "updates": UPDATES,
        "exploration_std": EXPLORATION_STD,
        "baseline_score": v202["development_score"],
        "history": history,
        "best_update": best["update"] if qualified else None,
        "status": "DEVELOPMENT_ONLINE_KINEMATIC_CANDIDATE_ONLY"
        if qualified
        else "REJECTED_ONLINE_KINEMATIC_PPO_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during online kinematic PPO")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "asset-root",
        "policy",
        "parent-report",
        "right-report",
        "refine-report",
        "lateral-report",
        "zero-report",
        "v202-report",
        "v203-report",
        "base-policy",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.zero_report,
        args.v202_report,
        args.v203_report,
        args.base_policy,
        args.output,
    )
    print(
        json.dumps(
            {key: report[key] for key in ("status", "baseline_score", "best_update", "report_hash")}
        )
    )


if __name__ == "__main__":
    main()
