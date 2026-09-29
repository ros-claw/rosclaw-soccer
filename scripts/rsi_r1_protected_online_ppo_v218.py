"""SIM_ONLY failure-prioritized online PPO behind hard-retained receiving skills."""

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
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_middle_basis_cem_v187 import PHYSICAL_KEYS
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy
from rsi_r1_protected_neural_fresh_v217 import FRESH_COURSES as CONSUMED_V217_COURSES

from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.rsi.receiving_protected_composed_neural import (
    ReceivingProtectedComposedNeural,
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

SCHEMA = "rosclaw_soccer.rsi.r1_protected_online_ppo_v218.result.v1"
SEED = 218929
UPDATES = 3
EXPLORATION_STD = 0.12


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
        protected,
        exploration_seed,
        exploration_std,
    ) = task
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingProtectedComposedNeural(
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
        protected_initial_features=protected,
        exploration_seed=exploration_seed,
        exploration_std=exploration_std,
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
        "protected_episode": feedback.protected_episode,
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
    v215_dir: Path,
    v216_report_path: Path,
    v217_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY protected online PPO evidence required")
    parent, right_parent, refine, lateral, v205, mapping, v214, v215, v216, v217 = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            v205_dir / "report.json",
            map_report_path,
            v214_report_path,
            v215_dir / "report.json",
            v216_report_path,
            v217_report_path,
        )
    )
    if (
        v217["status"] != "REJECTED_PROTECTED_NEURAL_FRESH_GATE"
        or v217["v216_report_hash"] != v216["report_hash"]
        or v216["v215_report_hash"] != v215["report_hash"]
        or v216["score"][:2] != [7, 16]
        or v217["candidate_score"][:2] != [4, 7]
        or not v216["protected_physical_equal"]
        or v205["map_report_hash"] != mapping["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed rejected fresh and six protected skills required")
    best = next(row for row in v215["history"] if row["update"] == v215["best_update"])
    initial = _load_policy(v215_dir / "update-3.npz", best["checkpoint_hash"])
    actor = KinematicActor(initial)
    reference = KinematicActor(initial)
    critic = torch.nn.Sequential(torch.nn.Linear(48, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.00015)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0004)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    rng = np.random.default_rng(SEED)
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
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
    slope = tuple(lateral["best"]["slope"])
    low_weights = tuple(float(value) for value in v214["low_weights"])
    protected = tuple(
        tuple(float(value) for value in row) for row in v216["protected_initial_features"]
    )
    consumed = (*CONSUMED_V207_COURSES, *CONSUMED_V214_COURSES, *CONSUMED_V217_COURSES)
    old_rows = (*v216["rows"], *v217["candidate"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_protected_online_ppo_v218.py",
            "src/rosclaw_soccer/rsi/receiving_protected_composed_neural.py",
            "src/rosclaw_soccer/rsi/receiving_composed_neural_residual.py",
            "scripts/rsi_r1_kinematic_online_ppo_v204.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)

    def tasks(
        courses: tuple[Any, ...], weights: Any, seeds: tuple[int, ...], std: float
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
                protected,
                seed,
                std,
            )
            for course, seed in zip(courses, seeds, strict=True)
        ]

    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        baseline = list(pool.map(_run_task, tasks(consumed, initial, (0,) * 24, 0.0)))
        baseline_equal = all(
            row["physical_trace_hash"] == old["physical_trace_hash"]
            for row, old in zip(baseline, old_rows, strict=True)
        )
        if not baseline_equal:
            raise ValueError("protected online start must exactly replay 24 sealed cases")
        hard_courses = tuple(
            course
            for course, row in zip(consumed, baseline, strict=True)
            if row["selected_expert"] != "parent"
            and not (clean(row) and row["controlled_reception"])
        )
        if not 8 <= len(hard_courses) <= 16:
            raise ValueError("nontrivial failure-prioritized curriculum required")
        history = []
        for update in range(UPDATES):
            sampled_courses = hard_courses * 3
            seeds = tuple(int(value) for value in rng.integers(0, 2**31, size=len(sampled_courses)))
            sampled = list(
                pool.map(_run_task, tasks(sampled_courses, actor.weights(), seeds, EXPLORATION_STD))
            )
            optimization = _update(
                actor, critic, actor_optimizer, critic_optimizer, sampled, reference
            )
            weights = actor.weights()
            checkpoint_hash = _save(output / f"update-{update + 1}.npz", weights)
            trajectories = []
            for index, row in enumerate(sampled):
                path = output / f"sample-u{update + 1}-e{index}.npz"
                np.savez_compressed(
                    path,
                    frames=np.asarray(row["observed_frames"], dtype=np.int64),
                    features=np.asarray(row["observed_features"], dtype=np.float64),
                    logits=np.asarray(row["sampled_logits"], dtype=np.float64),
                )
                trajectories.append(
                    {
                        "course": row["course"],
                        "safe": row["safe"],
                        "controlled_reception": row["controlled_reception"],
                        "own_nonfoot_frames": row["own_nonfoot_frames"],
                        "trace_hash": hash_bytes(path.read_bytes()),
                    }
                )
            deterministic = list(pool.map(_run_task, tasks(consumed, weights, (0,) * 24, 0.0)))
            score = _score(deterministic)
            retained = all(
                row["physical_trace_hash"] == old["physical_trace_hash"]
                for row, old in zip(deterministic, old_rows, strict=True)
                if row["course"]["seed"] in v216["protected_seeds"]
            )
            history.append(
                {
                    "update": update + 1,
                    "checkpoint_hash": checkpoint_hash,
                    "optimization": optimization,
                    "sample_score": _score(sampled),
                    "sample_trajectories": trajectories,
                    "deterministic_score": score,
                    "protected_physical_equal": retained,
                    "deterministic_rows": deterministic,
                }
            )
            print(
                json.dumps(
                    {
                        "update": update + 1,
                        "sample": _score(sampled),
                        "deterministic": score,
                        "protected": retained,
                    }
                ),
                flush=True,
            )
    best = max(history, key=lambda row: tuple(row["deterministic_score"]))
    qualified = (
        best["protected_physical_equal"]
        and best["deterministic_score"][0] >= 13
        and best["deterministic_score"][1] >= 23
    )
    report = {
        "schema": SCHEMA,
        "v217_report_hash": v217["report_hash"],
        "v216_report_hash": v216["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_FAILED_COURSES_ONLY_ONLINE_PPO_NOT_NEW_FRESH",
        "seed": SEED,
        "updates": UPDATES,
        "hard_course_seeds": [course.seed for course in hard_courses],
        "baseline_physical_equal": baseline_equal,
        "baseline_score": _score(baseline),
        "history": history,
        "best_update": best["update"],
        "status": "DEVELOPMENT_HARD_CURRICULUM_CANDIDATE_ONLY"
        if qualified
        else "REJECTED_PROTECTED_ONLINE_PPO_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during protected online PPO")
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
        "v215-dir",
        "v216-report",
        "v217-report",
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
        args.v215_dir,
        args.v216_report,
        args.v217_report,
        args.output,
    )
    print(
        json.dumps(
            {key: report[key] for key in ("status", "baseline_score", "best_update", "report_hash")}
        )
    )


if __name__ == "__main__":
    main()
