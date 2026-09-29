"""SIM_ONLY online actor-critic for measured-state eight-G1 receiving episodes."""

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
from rsi_r1_side_navigation_fresh_v160 import FRESH_COURSES, clean
from rsi_r1_taskspace_feedback_v125 import COURSES as ANCHOR_COURSES

from rosclaw_soccer.rsi.receiving_contextual_motor_expert import (
    ContextualMotorWeights,
    ReceivingContextualMotorExpert,
)
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.rsi.team_receive_side_navigation import TeamReceiveSideNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_contextual_actor_critic_v191.result.v1"
UPDATES = 3
EPISODES_PER_UPDATE = 36
NOISE_STD = 0.18


class ContextualActorCritic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 12),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1),
        )
        last = self.actor[-1]
        assert isinstance(last, torch.nn.Linear)
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)

    def weights(self) -> ContextualMotorWeights:
        first, last = self.actor[0], self.actor[-1]
        assert isinstance(first, torch.nn.Linear) and isinstance(last, torch.nn.Linear)
        return ContextualMotorWeights(
            tuple(float(v) for v in first.weight.detach().reshape(-1).cpu().numpy()),
            tuple(float(v) for v in first.bias.detach().cpu().numpy()),
            tuple(float(v) for v in last.weight.detach().reshape(-1).cpu().numpy()),
            tuple(float(v) for v in last.bias.detach().cpu().numpy()),
        )


def run_episode(
    asset_root: Path,
    policy_path: Path,
    course: ReceivingCourse,
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    slope: tuple[float, ...],
    weights: ContextualMotorWeights,
    noise: tuple[float, ...],
) -> dict[str, Any]:
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingContextualMotorExpert(
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
        exploration_noise=noise,
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
    ids = tuple(sorted(row["agent_id"] for row in info["qualities"]))
    _, outcome = receiving_window(
        trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
    )
    detail = explain_receiving_window(
        trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
    )
    code = ids.index(course.agent_id) + 1
    foot = np.asarray(trace["ball_contact_agent_code"])
    foot_force = np.asarray(trace["ball_contact_force_n"])
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    nonfoot_force = np.asarray(trace["ball_nonfoot_contact_force_n"])
    active = np.asarray(trace["receiving_impedance_active_substeps"], dtype=np.int64)
    return {
        "course": vars(course),
        "measured_context": feedback.measured_context,
        "raw_action": feedback.selected_raw_action,
        "action": feedback.selected_action,
        "active_substeps": int(active.sum()),
        "safe": info["safe"],
        "fault_agents": info["physics_evidence_fault_agents"],
        "first_foot_frame": next(
            (frame for frame in range(20, 120) if foot[frame] == code and foot_force[frame] > 0),
            None,
        ),
        "own_nonfoot_frames": [
            frame for frame in range(20, 120) if nonfoot[frame] == code and nonfoot_force[frame] > 0
        ],
        "controlled_reception": outcome["controlled_reception"],
        "tail_maximum_foot_distance_m": detail["tail_maximum_foot_distance_m"],
        "tail_maximum_ball_speed_mps": detail["tail_maximum_ball_speed_mps"],
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
    weights: ContextualMotorWeights,
    noises: tuple[tuple[float, ...], ...] | None = None,
) -> list[dict[str, Any]]:
    if noises is None:
        noises = ((0.0,) * 12,) * len(courses)
    jobs = [
        (asset_root, policy_path, course, coordination, left, right, slope, weights, noise)
        for course, noise in zip(courses, noises, strict=True)
    ]
    return list(pool.map(_run_task, jobs))


def reward(row: dict[str, Any]) -> float:
    if not row["safe"] or row["fault_agents"] or row["active_substeps"] > 32:
        return -60.0
    if row["first_foot_frame"] is None:
        return -25.0
    if row["own_nonfoot_frames"]:
        return -30.0 - 2.0 * len(row["own_nonfoot_frames"])
    distance = min(float(row["tail_maximum_foot_distance_m"]), 2.0)
    speed = min(float(row["tail_maximum_ball_speed_mps"]), 3.0)
    return 15.0 + 35.0 * float(row["controlled_reception"]) - 8.0 * distance - 6.0 * speed


def score(rows: list[dict[str, Any]]) -> tuple[int, int, int, float]:
    development, retention = rows[: len(COURSES)], rows[len(COURSES) :]
    return (
        int(all(clean(row) and row["controlled_reception"] for row in retention)),
        sum(clean(row) and row["controlled_reception"] for row in development),
        sum(clean(row) for row in development),
        float(sum(reward(row) for row in development)),
    )


def _behavior_cloning(
    model: ContextualActorCritic, baseline: list[dict[str, Any]], map_report: dict[str, Any]
) -> dict[str, Any]:
    contexts: list[tuple[float, float]] = []
    targets: list[list[float]] = []
    labels: list[dict[str, Any]] = []
    expert_weights = {
        "parent": (0.0,) * 12,
        "low": tuple(map_report["selected_weights"]["low"]),
        "center": tuple(map_report["selected_weights"]["center"]),
        "high": tuple(map_report["selected_weights"]["high"]),
    }
    for index, row in enumerate(baseline[: len(COURSES)]):
        group = map_report["rows"][index * 4 : index * 4 + 4]
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
        name = teacher["expert"]
        feature = float(teacher["summary"]["middle_feature"] or 0.0)
        contexts.append(tuple(row["measured_context"]))
        targets.append([feature * value for value in expert_weights[name]])
        labels.append({"course": row["course"], "expert": name, "feature": feature})
    for row in baseline[len(COURSES) :]:
        contexts.append(tuple(row["measured_context"]))
        targets.append([0.0] * 12)
        labels.append({"course": row["course"], "expert": "retention_zero"})
    if len(contexts) < 4:
        raise ValueError("insufficient positive measured imitation labels")
    states = torch.tensor(contexts, dtype=torch.float32)
    actions = torch.tensor(targets, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=0.003)
    for _ in range(300):
        predicted = model.actor(states).tanh()
        loss = torch.nn.functional.mse_loss(predicted, actions) + 0.01 * predicted.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
        optimizer.step()
    return {"labels": labels, "loss": float(loss.detach())}


def _ppo_update(
    model: ContextualActorCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    samples: list[dict[str, Any]],
) -> dict[str, float]:
    states = torch.tensor([row["measured_context"] for row in samples], dtype=torch.float32)
    actions = torch.tensor([row["raw_action"] for row in samples], dtype=torch.float32)
    rewards = torch.tensor([reward(row) / 50.0 for row in samples], dtype=torch.float32)
    with torch.no_grad():
        old_mean = model.actor(states)
        old_logp = torch.distributions.Normal(old_mean, NOISE_STD).log_prob(actions).sum(-1)
        advantages = rewards - model.critic(states).squeeze(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
    actor_loss = torch.tensor(0.0)
    critic_loss = torch.tensor(0.0)
    for _ in range(8):
        mean = model.actor(states)
        new_logp = torch.distributions.Normal(mean, NOISE_STD).log_prob(actions).sum(-1)
        ratio = (new_logp - old_logp).exp()
        clipped = ratio.clamp(0.8, 1.2)
        actor_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
        actor_loss = actor_loss + 0.005 * mean.square().mean()
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5)
        actor_optimizer.step()
        value = model.critic(states).squeeze(-1)
        critic_loss = torch.nn.functional.mse_loss(value, rewards)
        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 0.5)
        critic_optimizer.step()
    return {"actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach())}


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
    output: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root) or not 0 <= seed < 2**31:
        raise ValueError("new external SIM_ONLY contextual actor-critic evidence required")
    course_bank = (*COURSES, FRESH_COURSES[0], ANCHOR_COURSES[0])
    bank_hash = preflight_receiving_courses(course_bank)
    parent, right_parent, refine, lateral, v187, v188, mapping = (
        json.loads(path.read_text())
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            v187_report,
            v188_report,
            map_report_path,
        )
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, refine, lateral, v187, v188, mapping)
    ) or (
        mapping["status"] != "INSUFFICIENT_SPECIALIST_COVERAGE"
        or mapping["courses"] != [vars(course) for course in COURSES]
        or mapping["oracle_coverage"] != 6
        or v188["previous_report_hash"] != v187["report_hash"]
        or mapping["v189_report_hash"] is None
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed failed continuous course map required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_contextual_actor_critic_v191.py",
            "scripts/rsi_r1_middle_basis_cem_v187.py",
            "src/rosclaw_soccer/rsi/receiving_contextual_motor_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ContextualActorCritic()
    output.mkdir(parents=True)
    context = multiprocessing.get_context("spawn")
    history: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
        baseline = evaluate(
            pool,
            asset_root,
            policy_path,
            course_bank,
            coordination,
            left,
            right,
            slope,
            model.weights(),
        )
        if any(
            row["physical_trace_hash"] != old["summary"]["physical_trace_hash"]
            for row, old in zip(baseline[: len(COURSES)], mapping["rows"][::4], strict=True)
        ):
            raise ValueError("zero neural actor failed exact 16-course physical equivalence")
        zero_weights = model.weights()
        selected_weights = {
            "low": tuple(_find_candidate(v188, 1, 6)["middle_weights"]),
            "center": tuple(_find_candidate(v188, 2, 7)["middle_weights"]),
            "high": tuple(_find_candidate(v187, 2, 2)["middle_weights"]),
        }
        mapping["selected_weights"] = selected_weights
        cloning = _behavior_cloning(model, baseline, mapping)
        imitation = evaluate(
            pool,
            asset_root,
            policy_path,
            course_bank,
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
            best_weights = zero_weights
            best_stage = "zero_parent"
        actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=0.0007)
        critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=0.001)
        for update in range(UPDATES):
            sampled_courses = (
                *COURSES,
                *COURSES,
                FRESH_COURSES[0],
                ANCHOR_COURSES[0],
                FRESH_COURSES[0],
                ANCHOR_COURSES[0],
            )
            assert len(sampled_courses) == EPISODES_PER_UPDATE
            noises = tuple(
                tuple(float(value) for value in np.clip(rng.normal(0.0, NOISE_STD, size=12), -1, 1))
                for _ in sampled_courses
            )
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
                noises,
            )
            optimization = _ppo_update(model, actor_optimizer, critic_optimizer, sampled)
            deterministic = evaluate(
                pool,
                asset_root,
                policy_path,
                course_bank,
                coordination,
                left,
                right,
                slope,
                model.weights(),
            )
            if score(deterministic) > score(best_rows):
                best_rows = deterministic
                best_weights = model.weights()
                best_stage = f"update-{update + 1}"
            row = {
                "update": update + 1,
                "sample_count": len(sampled),
                "sampled_clean": sum(clean(item) for item in sampled),
                "sampled_controlled": sum(
                    clean(item) and item["controlled_reception"] for item in sampled
                ),
                "mean_reward": float(np.mean([reward(item) for item in sampled])),
                "deterministic_score": score(deterministic),
                "optimization": optimization,
            }
            history.append(row)
            (output / "progress.json").write_text(json.dumps(history, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    weights_path = output / "best-policy.npz"
    np.savez_compressed(
        weights_path,
        input_matrix=np.asarray(best_weights.input_matrix),
        input_bias=np.asarray(best_weights.input_bias),
        output_matrix=np.asarray(best_weights.output_matrix),
        output_bias=np.asarray(best_weights.output_bias),
    )
    qualified = score(best_rows)[0] == 1 and score(best_rows)[1] >= 12
    report = {
        "schema": SCHEMA,
        "map_report_hash": mapping["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_EIGHT_G1_CONTEXTUAL_ONLINE_ACTOR_CRITIC",
        "course_bank_hash": bank_hash,
        "course_count": len(course_bank),
        "seed": seed,
        "updates": UPDATES,
        "episodes_per_update": EPISODES_PER_UPDATE,
        "noise_std": NOISE_STD,
        "zero_physics_equal": True,
        "baseline_score": score(baseline),
        "imitation": cloning,
        "imitation_score": score(imitation),
        "history": history,
        "best_stage": best_stage,
        "best_score": score(best_rows),
        "best_rows": best_rows,
        "policy_hash": hash_bytes(weights_path.read_bytes()),
        "status": "DEVELOPMENT_CONTEXTUAL_BASIN_UNVALIDATED"
        if qualified
        else "REJECTED_CONTEXTUAL_ACTOR_CRITIC_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during contextual online actor-critic")
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=191930)
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
