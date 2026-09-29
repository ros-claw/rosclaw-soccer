"""SIM_ONLY online 50 Hz actor-critic on consumed native eight-G1 courses."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsi_r1_course_map_v190 import COURSES, _find_candidate
from rsi_r1_middle_basis_cem_v187 import PHYSICAL_KEYS
from rsi_r1_temporal_zero_v192 import TRAIN_COURSES

from rosclaw_soccer.rsi.receiving_temporal_motor_expert import (
    ReceivingTemporalMotorExpert,
    TemporalMotorWeights,
)
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.rsi.team_receive_side_navigation import TeamReceiveSideNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_temporal_actor_critic_v193.result.v1"
UPDATES = 4
EPISODES_PER_UPDATE = 36
EXPLORATION_STD = 0.12


def run_episode(
    asset_root: Path,
    policy_path: Path,
    course: ReceivingCourse,
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    slope: tuple[float, ...],
    weights: TemporalMotorWeights,
    exploration_std: float,
    exploration_seed: int,
) -> dict[str, Any]:
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingTemporalMotorExpert(
        course.agent_id,
        schedule.contract_hash,
        mailbox,
        coordination,
        left,
        right,
        left_post_gain=0.0,
        right_post_gain=0.4,
        old_lateral_slope=slope,
        far_lateral_slope=(0.0,) * 12,
        policy=weights,
        exploration_std=exploration_std,
        exploration_seed=exploration_seed,
    )
    navigation = TeamReceiveSideNavigation(
        course.agent_id,
        hash_bytes((asset_root / "policy/loco_mode/model/policy_29dof.pt").read_bytes()),
        hash_bytes((asset_root / "policy/loco_mode/config/LocoMode.yaml").read_bytes()),
        mailbox,
        (0.0, 0.0, 0.5, 0.0, 0.0),
        (0.0,) * 5,
    )
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy_path,
        course=course,
        scenario_id=f"s199.rsi.r1.coordinated-receiving.{course.seed}",
        configuration_profile="R1_CONTACT_TAP",
        oracle=schedule,
        feedback_provider=feedback,
        research_navigation_policy=navigation,
        research_contact_leg_stiffness_scale=0.4,
        research_contact_distance_threshold_m=0.24,
        research_contact_max_active_substeps=32,
        physics_evidence_consumers={course.agent_id: ReceiveContactEvidence(mailbox)},
    )
    info = result.to_dict()
    n = len(trace["time"])
    ids = tuple(sorted(row["agent_id"] for row in info["qualities"]))
    code = ids.index(course.agent_id) + 1
    foot = np.asarray(trace["ball_contact_agent_code"])
    foot_force = np.asarray(trace["ball_contact_force_n"])
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    nonfoot_force = np.asarray(trace["ball_nonfoot_contact_force_n"])
    active = np.asarray(trace["receiving_impedance_active_substeps"], dtype=np.int64)
    if n >= 120:
        shaped, outcome = receiving_window(
            trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
        )
        detail = explain_receiving_window(
            trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
        )
        distance = detail["tail_maximum_foot_distance_m"]
        speed = detail["tail_maximum_ball_speed_mps"]
    else:
        shaped = np.asarray([], dtype=np.float32)
        outcome = {"controlled_reception": False}
        distance = 2.0
        speed = 3.0
    return {
        "course": vars(course),
        "observed_frames": feedback.observed_frames,
        "observed_features": feedback.observed_features,
        "sampled_logits": feedback.sampled_logits,
        "shaped_reward": [float(value) for value in shaped],
        "frames_recorded": n,
        "early_termination": n < 120,
        "active_substeps": int(active.sum()),
        "safe": info["safe"] and n >= 120,
        "fault_agents": info["physics_evidence_fault_agents"],
        "first_foot_frame": next(
            (
                frame
                for frame in range(20, min(n, 120))
                if foot[frame] == code and foot_force[frame] > 0
            ),
            None,
        ),
        "own_nonfoot_frames": [
            frame
            for frame in range(20, min(n, 120))
            if nonfoot[frame] == code and nonfoot_force[frame] > 0
        ],
        "controlled_reception": outcome["controlled_reception"],
        "tail_maximum_foot_distance_m": distance,
        "tail_maximum_ball_speed_mps": speed,
        "physical_trace_hash": hash_json(
            {key: np.asarray(trace[key]).tolist() for key in PHYSICAL_KEYS}
        ),
    }


def _run_task(args: tuple[Any, ...]) -> dict[str, Any]:
    return run_episode(*args)


def evaluate(
    pool: ProcessPoolExecutor,
    asset_root: Path,
    policy_path: Path,
    courses: tuple[ReceivingCourse, ...],
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    slope: tuple[float, ...],
    weights: TemporalMotorWeights,
    exploration_std: float = 0.0,
    seeds: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    if seeds is None:
        seeds = tuple(0 for _ in courses)
    tasks = [
        (
            asset_root,
            policy_path,
            course,
            coordination,
            left,
            right,
            slope,
            weights,
            exploration_std,
            seed,
        )
        for course, seed in zip(courses, seeds, strict=True)
    ]
    return list(pool.map(_run_task, tasks))


class TemporalActorCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(10, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 12),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(10, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1),
        )
        last = self.actor[-1]
        assert isinstance(last, torch.nn.Linear)
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)

    def weights(self) -> TemporalMotorWeights:
        first, last = self.actor[0], self.actor[-1]
        assert isinstance(first, torch.nn.Linear) and isinstance(last, torch.nn.Linear)
        return TemporalMotorWeights(
            tuple(float(v) for v in first.weight.detach().reshape(-1).cpu().numpy()),
            tuple(float(v) for v in first.bias.detach().cpu().numpy()),
            tuple(float(v) for v in last.weight.detach().reshape(-1).cpu().numpy()),
            tuple(float(v) for v in last.bias.detach().cpu().numpy()),
        )


def clean(row: dict[str, Any]) -> bool:
    return bool(
        row["safe"]
        and not row["fault_agents"]
        and row["first_foot_frame"] is not None
        and not row["own_nonfoot_frames"]
        and row["active_substeps"] <= 32
    )


def reward(row: dict[str, Any]) -> float:
    if not row["safe"] or row["fault_agents"] or row["active_substeps"] > 32:
        return -60.0
    if row["first_foot_frame"] is None:
        return -25.0
    if row["own_nonfoot_frames"]:
        return -35.0 - 2.0 * len(row["own_nonfoot_frames"])
    distance = min(float(row["tail_maximum_foot_distance_m"]), 2.0)
    speed = min(float(row["tail_maximum_ball_speed_mps"]), 3.0)
    return 18.0 + 50.0 * float(row["controlled_reception"]) - 10.0 * distance - 8.0 * speed


def score(rows: list[dict[str, Any]]) -> tuple[int, int, int, float]:
    development, retention = rows[: len(COURSES)], rows[len(COURSES) :]
    return (
        int(all(clean(row) and row["controlled_reception"] for row in retention)),
        sum(clean(row) and row["controlled_reception"] for row in development),
        sum(clean(row) for row in development),
        float(sum(reward(row) for row in development)),
    )


def _old_fraction(frame: int) -> float:
    if frame < 15:
        return 0.0
    if frame < 19:
        return (frame - 15) / 4.0
    if frame <= 30:
        return 1.0
    if frame < 40:
        return (40 - frame) / 10.0
    return 0.0


def _new_fraction(frame: int) -> float:
    if frame < 19:
        return (frame - 15) / 4.0
    if frame <= 50:
        return 1.0
    return (65 - frame) / 15.0


def behavior_clone(
    model: TemporalActorCritic,
    zero: list[dict[str, Any]],
    mapping: dict[str, Any],
    v187: dict[str, Any],
    v188: dict[str, Any],
) -> dict[str, Any]:
    expert_weights = {
        "parent": (0.0,) * 12,
        "low": tuple(_find_candidate(v188, 1, 6)["middle_weights"]),
        "center": tuple(_find_candidate(v188, 2, 7)["middle_weights"]),
        "high": tuple(_find_candidate(v187, 2, 2)["middle_weights"]),
    }
    features: list[tuple[float, ...]] = []
    targets: list[list[float]] = []
    sample_weights: list[float] = []
    labels: list[dict[str, Any]] = []
    for index, row in enumerate(zero[: len(COURSES)]):
        group = mapping["rows"][index * 4 : index * 4 + 4]
        positive = [
            item
            for item in group
            if clean(item["summary"]) and item["summary"]["controlled_reception"]
        ]
        if not positive:
            continue
        teacher = min(
            positive,
            key=lambda item: (
                item["summary"]["tail_maximum_foot_distance_m"]
                + item["summary"]["tail_maximum_ball_speed_mps"]
            ),
        )
        expert = teacher["expert"]
        basis = float(teacher["summary"]["middle_feature"] or 0.0)
        labels.append({"course": row["course"], "expert": expert, "basis": basis})
        for frame, observation in zip(
            row["observed_frames"], row["observed_features"], strict=True
        ):
            fraction = _old_fraction(frame) / max(_new_fraction(frame), 1e-6)
            features.append(tuple(observation))
            targets.append([fraction * basis * value for value in expert_weights[expert]])
            sample_weights.append(1.0)
    for row in zero[len(COURSES) :]:
        labels.append({"course": row["course"], "expert": "retention_zero"})
        for observation in row["observed_features"]:
            features.append(tuple(observation))
            targets.append([0.0] * 12)
            sample_weights.append(5.0)
    if len(labels) < 4:
        raise ValueError("insufficient temporal imitation labels")
    inputs = torch.tensor(features, dtype=torch.float32)
    expected = torch.tensor(targets, dtype=torch.float32)
    weights = torch.tensor(sample_weights, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=0.002)
    for _ in range(400):
        predicted = model.actor(inputs).tanh()
        loss = (weights[:, None] * (predicted - expected).square()).mean()
        loss = loss + 0.005 * predicted.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
        optimizer.step()
    return {"labels": labels, "frame_labels": len(features), "loss": float(loss.detach())}


def ppo_update(
    model: TemporalActorCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    episodes: list[dict[str, Any]],
    anchor_features: torch.Tensor,
) -> dict[str, float]:
    states = torch.tensor(
        [state for episode in episodes for state in episode["observed_features"]],
        dtype=torch.float32,
    )
    actions = torch.tensor(
        [action for episode in episodes for action in episode["sampled_logits"]],
        dtype=torch.float32,
    )
    returns = torch.tensor(
        [
            reward(episode) / 50.0 + sum(episode["shaped_reward"]) / 10.0
            for episode in episodes
            for _ in episode["observed_frames"]
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        old_mean = model.actor(states)
        old_logp = torch.distributions.Normal(old_mean, EXPLORATION_STD).log_prob(actions).sum(-1)
        advantages = returns - model.critic(states).squeeze(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    actor_loss = torch.tensor(0.0)
    critic_loss = torch.tensor(0.0)
    for _ in range(6):
        for indices in torch.randperm(len(states)).split(512):
            mean = model.actor(states[indices])
            new_logp = (
                torch.distributions.Normal(mean, EXPLORATION_STD).log_prob(actions[indices]).sum(-1)
            )
            ratio = (new_logp - old_logp[indices]).exp()
            actor_loss = -torch.minimum(
                ratio * advantages[indices], ratio.clamp(0.8, 1.2) * advantages[indices]
            ).mean()
            # Stability-plasticity: anchor output is constrained during every update.
            anchor_penalty = model.actor(anchor_features).tanh().square().mean()
            actor_loss = actor_loss + 0.5 * anchor_penalty + 0.005 * mean.square().mean()
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5)
            actor_optimizer.step()
            critic_loss = torch.nn.functional.mse_loss(
                model.critic(states[indices]).squeeze(-1), returns[indices]
            )
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 0.5)
            critic_optimizer.step()
    return {
        "actor_loss": float(actor_loss.detach()),
        "critic_loss": float(critic_loss.detach()),
        "anchor_penalty": float(anchor_penalty.detach()),
    }


def _save_weights(path: Path, weights: TemporalMotorWeights) -> str:
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
    v187_report: Path,
    v188_report: Path,
    map_report_path: Path,
    zero_report_path: Path,
    output: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root) or not 0 <= seed < 2**31:
        raise ValueError("new external SIM_ONLY temporal actor-critic evidence required")
    parent, right_parent, refine, lateral, v187, v188, mapping, zero = (
        json.loads(path.read_text())
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            v187_report,
            v188_report,
            map_report_path,
            zero_report_path,
        )
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, refine, lateral, v187, v188, mapping, zero)
    ) or (
        zero["status"] != "TEMPORAL_ZERO_QUALIFIED_FOR_TRAINING"
        or zero["map_report_hash"] != mapping["report_hash"]
        or not zero["zero_physics_equal"]
        or mapping["oracle_coverage"] != 6
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed temporal zero and development map required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_temporal_actor_critic_v193.py",
            "scripts/rsi_r1_temporal_zero_v192.py",
            "src/rosclaw_soccer/rsi/receiving_temporal_motor_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    model = TemporalActorCritic()
    output.mkdir(parents=True)
    baseline = zero["rows"]
    cloning = behavior_clone(model, baseline, mapping, v187, v188)
    context = multiprocessing.get_context("spawn")
    history: list[dict[str, Any]] = []
    anchor_features = torch.tensor(
        [feature for row in baseline[len(COURSES) :] for feature in row["observed_features"]],
        dtype=torch.float32,
    )
    with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
        imitation = evaluate(
            pool,
            asset_root,
            policy_path,
            TRAIN_COURSES,
            coordination,
            left,
            right,
            slope,
            model.weights(),
        )
        if score(imitation) > score(baseline):
            best_rows = imitation
            best_weights = model.weights()
            best_stage = "imitation"
        else:
            best_rows = baseline
            best_weights = TemporalMotorWeights()
            best_stage = "zero_parent"
        actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=0.0003)
        critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=0.0008)
        for update in range(UPDATES):
            sampled_courses = (
                *COURSES,
                *COURSES,
                TRAIN_COURSES[-2],
                TRAIN_COURSES[-1],
                TRAIN_COURSES[-2],
                TRAIN_COURSES[-1],
            )
            assert len(sampled_courses) == EPISODES_PER_UPDATE
            seeds = tuple(int(value) for value in rng.integers(0, 2**31, size=len(sampled_courses)))
            sampled = evaluate(
                pool,
                asset_root,
                policy_path,
                sampled_courses,
                coordination,
                left,
                right,
                slope,
                model.weights(),
                EXPLORATION_STD,
                seeds,
            )
            optimization = ppo_update(
                model, actor_optimizer, critic_optimizer, sampled, anchor_features
            )
            deterministic = evaluate(
                pool,
                asset_root,
                policy_path,
                TRAIN_COURSES,
                coordination,
                left,
                right,
                slope,
                model.weights(),
            )
            checkpoint_hash = _save_weights(output / f"update-{update + 1}.npz", model.weights())
            if score(deterministic) > score(best_rows):
                best_rows = deterministic
                best_weights = model.weights()
                best_stage = f"update-{update + 1}"
            item = {
                "update": update + 1,
                "episodes": len(sampled),
                "sampled_clean": sum(clean(row) for row in sampled),
                "sampled_controlled": sum(
                    clean(row) and row["controlled_reception"] for row in sampled
                ),
                "mean_reward": float(np.mean([reward(row) for row in sampled])),
                "deterministic_score": score(deterministic),
                "checkpoint_hash": checkpoint_hash,
                "optimization": optimization,
            }
            history.append(item)
            (output / "progress.json").write_text(json.dumps(history, indent=2) + "\n")
            print(json.dumps(item), flush=True)
    best_policy_hash = _save_weights(output / "best-policy.npz", best_weights)
    qualified = score(best_rows)[0] == 1 and score(best_rows)[1] >= 12
    report = {
        "schema": SCHEMA,
        "zero_report_hash": zero["report_hash"],
        "map_report_hash": mapping["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_EIGHT_G1_50HZ_ONLINE_ACTOR_CRITIC",
        "seed": seed,
        "updates": UPDATES,
        "episodes_per_update": EPISODES_PER_UPDATE,
        "exploration_std": EXPLORATION_STD,
        "zero_physics_equal": True,
        "baseline_score": score(baseline),
        "imitation": cloning,
        "imitation_score": score(imitation),
        "history": history,
        "best_stage": best_stage,
        "best_score": score(best_rows),
        "best_rows": best_rows,
        "best_policy_hash": best_policy_hash,
        "status": "DEVELOPMENT_TEMPORAL_BASIN_UNVALIDATED"
        if qualified
        else "REJECTED_TEMPORAL_ACTOR_CRITIC_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during temporal online actor-critic")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--refine-report", type=Path, required=True)
    parser.add_argument("--lateral-report", type=Path, required=True)
    parser.add_argument("--v187-report", type=Path, required=True)
    parser.add_argument("--v188-report", type=Path, required=True)
    parser.add_argument("--map-report", type=Path, required=True)
    parser.add_argument("--zero-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=193930)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.v187_report,
        args.v188_report,
        args.map_report,
        args.zero_report,
        args.output,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "baseline_score": report["baseline_score"],
                "imitation_score": report["imitation_score"],
                "best_score": report["best_score"],
                "report_hash": report["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
