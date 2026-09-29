"""SIM_ONLY online 48D PPO residual over a frozen composed receiving parent."""

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
from rsi_r1_composed_contact_fresh_v214 import FRESH_COURSES as CONSUMED_V214_COURSES
from rsi_r1_kinematic_advantage_v201 import KinematicActor
from rsi_r1_kinematic_online_ppo_v204 import _save, _update
from rsi_r1_measured_skill_router_v207 import FRESH_COURSES as CONSUMED_V207_COURSES
from rsi_r1_measured_skill_router_v207 import _score
from rsi_r1_middle_basis_cem_v187 import PHYSICAL_KEYS
from rsi_r1_temporal_zero_v192 import TRAIN_COURSES

from rosclaw_soccer.rsi.receiving_composed_neural_residual import ReceivingComposedNeuralResidual
from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import KinematicMotorWeights
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.rsi.team_receive_side_navigation import TeamReceiveSideNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window

SCHEMA = "rosclaw_soccer.rsi.r1_composed_neural_ppo_v215.result.v1"
UPDATES = 3
EXPLORATION_STD = 0.12
SEED = 215929


def _run_task(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        asset_root,
        policy_path,
        course,
        coordination,
        left,
        right,
        slope,
        knots,
        references,
        low_weights,
        weights,
        exploration_std,
        exploration_seed,
    ) = task
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingComposedNeuralResidual(
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
        skill_knots=knots,
        references=references,
        corrected_experts=("high",),
        foot_gain=0.3,
        post_multiplier=1.0,
        low_post_weights=low_weights,
        neural_weights=weights,
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
    ids = tuple(sorted(row["agent_id"] for row in info["qualities"]))
    n = len(trace["time"])
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
        shaped = np.asarray([], dtype=np.float64)
        outcome = {"controlled_reception": False}
        distance = 2.0
        speed = 3.0
    code = ids.index(course.agent_id) + 1
    foot = np.asarray(trace["ball_contact_agent_code"])
    foot_force = np.asarray(trace["ball_contact_force_n"])
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    nonfoot_force = np.asarray(trace["ball_nonfoot_contact_force_n"])
    active = np.asarray(trace["receiving_impedance_active_substeps"], dtype=np.int64)
    return {
        "course": vars(course),
        "selected_expert": feedback.selected_expert,
        "observed_frames": feedback.observed_frames,
        "observed_features": feedback.observed_features,
        "sampled_logits": feedback.sampled_logits,
        "shaped_reward": [float(value) for value in shaped],
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
        "active_substeps": int(active.sum()),
        "controlled_reception": outcome["controlled_reception"],
        "tail_maximum_foot_distance_m": distance,
        "tail_maximum_ball_speed_mps": speed,
        "physical_trace_hash": hash_json(
            {key: np.asarray(trace[key]).tolist() for key in PHYSICAL_KEYS}
        ),
    }


def train(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    v205_dir: Path,
    map_report_path: Path,
    v214_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY composed neural PPO evidence required")
    parent, right_parent, refine, lateral, v205, mapping, v214 = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            v205_dir / "report.json",
            map_report_path,
            v214_report_path,
        )
    )
    if (
        v214["status"] != "REJECTED_COMPOSED_CONTACT_FRESH_GATE"
        or v214["development_score"][:2] != [4, 8]
        or v214["fresh_candidate_score"][:2] != [2, 6]
        or not v214["anchor_physical_equal"]
        or v205["map_report_hash"] != mapping["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed failed composed fresh and verified teachers required")
    knots = tuple(
        ReceivingSkillKnot(
            float(row["lateral_m"]),
            float(row["closing_speed_mps"]),
            row["expert"],
            tuple(float(value) for value in row["weights"]),
        )
        for row in v214["knots"]
    )
    references = []
    for index, row in enumerate(v205["rows"]):
        if row["expert"] != "high":
            continue
        path = v205_dir / f"teacher-{index}.npz"
        if hash_bytes(path.read_bytes()) != row["feature_hash"]:
            raise ValueError("sealed physical high teacher required")
        with np.load(path, allow_pickle=False) as arrays:
            features = np.asarray(arrays["features"], dtype=np.float64)
        historical = next(
            item["summary"]
            for item in mapping["rows"]
            if item["expert"] == "high"
            and item["summary"]["course"]["seed"] == row["course"]["seed"]
        )
        references.append(
            ReceivingFootPhaseReference(
                "high",
                float(features[0, 1] * 0.2),
                float(features[0, 3] * 2.0),
                int(historical["first_foot_frame"]),
                tuple(tuple(float(value) for value in frame) for frame in features),
            )
        )
    low_weights = tuple(float(value) for value in v214["low_weights"])
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_composed_neural_ppo_v215.py",
            "src/rosclaw_soccer/rsi/receiving_composed_neural_residual.py",
            "src/rosclaw_soccer/rsi/receiving_composed_contact_router.py",
            "scripts/rsi_r1_kinematic_online_ppo_v204.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
    first = tuple(float(value) for value in rng.normal(0.0, 0.04, 48 * 32))
    base_weights = KinematicMotorWeights(input_matrix=first)
    actor = KinematicActor(base_weights)
    reference = KinematicActor(base_weights)
    critic = torch.nn.Sequential(torch.nn.Linear(48, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0002)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0005)
    consumed = (*CONSUMED_V207_COURSES, *CONSUMED_V214_COURSES)

    def tasks(
        courses: tuple[Any, ...], weights: KinematicMotorWeights, std: float, seeds: tuple[int, ...]
    ) -> list[tuple[Any, ...]]:
        return [
            (
                asset_root,
                policy_path,
                course,
                coordination,
                left,
                right,
                slope,
                knots,
                tuple(references),
                low_weights,
                weights,
                std,
                seed,
            )
            for course, seed in zip(courses, seeds, strict=True)
        ]

    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        baseline = list(pool.map(_run_task, tasks(consumed, base_weights, 0.0, (0,) * 16)))
        baseline_equal = all(
            row["physical_trace_hash"] == old["physical_trace_hash"]
            for row, old in zip(
                baseline, (*v214["development"], *v214["fresh_candidate"]), strict=True
            )
        )
        if not baseline_equal:
            raise ValueError("zero neural residual must exactly replay composed physical parent")
        anchors = list(pool.map(_run_task, tasks(TRAIN_COURSES[-2:], base_weights, 0.0, (0, 0))))
        if any(
            row["physical_trace_hash"] != old["physical_trace_hash"]
            for row, old in zip(anchors, v214["anchors"], strict=True)
        ):
            raise ValueError("old parent anchor must remain physically protected")
        history = []
        for update in range(UPDATES):
            sampled_courses = consumed * 2
            seeds = tuple(int(value) for value in rng.integers(0, 2**31, size=32))
            sampled = list(
                pool.map(_run_task, tasks(sampled_courses, actor.weights(), EXPLORATION_STD, seeds))
            )
            optimization = _update(
                actor, critic, actor_optimizer, critic_optimizer, sampled, reference
            )
            weights = actor.weights()
            checkpoint_hash = _save(output / f"update-{update + 1}.npz", weights)
            deterministic = list(pool.map(_run_task, tasks(consumed, weights, 0.0, (0,) * 16)))
            score = _score(deterministic)
            sample_score = _score(sampled)
            history.append(
                {
                    "update": update + 1,
                    "checkpoint_hash": checkpoint_hash,
                    "optimization": optimization,
                    "sample_score": sample_score,
                    "deterministic_score": score,
                    "deterministic_rows": deterministic,
                }
            )
            print(
                json.dumps({"update": update + 1, "sample": sample_score, "deterministic": score}),
                flush=True,
            )
    best = max(history, key=lambda row: tuple(row["deterministic_score"]))
    development_qualified = (
        best["deterministic_score"][0] >= 8 and best["deterministic_score"][1] >= 14
    )
    report = {
        "schema": SCHEMA,
        "v214_report_hash": v214["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_V207_AND_V214_ONLINE_48D_PPO_NO_NEW_FRESH",
        "seed": SEED,
        "updates": UPDATES,
        "exploration_std": EXPLORATION_STD,
        "base_policy_hash": base_weights.contract_hash,
        "baseline_physical_equal": baseline_equal,
        "baseline_score": _score(baseline),
        "history": history,
        "best_update": best["update"],
        "development_qualified": development_qualified,
        "status": "DEVELOPMENT_NEURAL_RESIDUAL_CANDIDATE_ONLY"
        if development_qualified
        else "REJECTED_COMPOSED_NEURAL_PPO_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during composed neural PPO")
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
        "v205-dir",
        "map-report",
        "v214-report",
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
        args.v205_dir,
        args.map_report,
        args.v214_report,
        args.output,
    )
    print(
        json.dumps(
            {key: report[key] for key in ("status", "baseline_score", "best_update", "report_hash")}
        )
    )


if __name__ == "__main__":
    main()
