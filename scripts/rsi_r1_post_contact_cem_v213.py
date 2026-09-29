"""SIM_ONLY online CEM of a measured-touch support skill in native eight-G1 physics."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_measured_skill_router_v207 import FRESH_COURSES, _score, clean
from rsi_r1_middle_basis_cem_v187 import PHYSICAL_KEYS
from rsi_r1_temporal_zero_v192 import TRAIN_COURSES

from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.rsi.receiving_post_contact_residual import ReceivingPostContactResidual
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.rsi.team_receive_side_navigation import TeamReceiveSideNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window

SCHEMA = "rosclaw_soccer.rsi.r1_post_contact_cem_v213.result.v1"
TRAIN_INDICES = (0, 2, 3, 6)
SEED = 213929


def _run_task(task: tuple[Any, ...]) -> dict[str, Any]:
    asset_root, policy_path, course, coordination, left, right, slope, knots, weights = task
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingPostContactResidual(
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
        post_weights=weights,
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
        _, outcome = receiving_window(
            trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
        )
        detail = explain_receiving_window(
            trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
        )
        distance = detail["tail_maximum_foot_distance_m"]
        speed = detail["tail_maximum_ball_speed_mps"]
    else:
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
        "active_substeps": int(active.sum()),
        "tail_maximum_foot_distance_m": distance,
        "tail_maximum_ball_speed_mps": speed,
        "active_frames": feedback.active_frames,
        "peak_residual_rad": feedback.peak_residual_rad,
        "physical_trace_hash": hash_json(
            {key: np.asarray(trace[key]).tolist() for key in PHYSICAL_KEYS}
        ),
    }


def _rank(rows: list[dict[str, Any]], weights: tuple[float, ...]) -> tuple[int, int, float, float]:
    return (
        sum(clean(row) for row in rows),
        sum(clean(row) and row["controlled_reception"] for row in rows),
        -sum(
            row["tail_maximum_foot_distance_m"] + row["tail_maximum_ball_speed_mps"] for row in rows
        ),
        -float(np.linalg.norm(weights)),
    )


def train(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    router_report_path: Path,
    v212_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY post-contact CEM evidence required")
    parent, right_parent, refine, lateral, router, v212 = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            router_report_path,
            v212_report_path,
        )
    )
    if (
        v212["status"] != "INSUFFICIENT_MOTOR_SKILL_COVERAGE"
        or v212["router_report_hash"] != router["report_hash"]
        or v212["oracle_coverage"] != 2
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed insufficient motor skill coverage required")
    knots = tuple(
        ReceivingSkillKnot(
            float(row["lateral_m"]),
            float(row["closing_speed_mps"]),
            row["expert"],
            tuple(float(value) for value in row["weights"]),
        )
        for row in router["skill_knots"]
    )
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_post_contact_cem_v213.py",
            "src/rosclaw_soccer/rsi/receiving_post_contact_residual.py",
            "src/rosclaw_soccer/rsi/receiving_measured_skill_router.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    zero = (0.0,) * 12

    def tasks(courses: tuple[Any, ...], weights: tuple[float, ...]) -> list[tuple[Any, ...]]:
        return [
            (asset_root, policy_path, course, coordination, left, right, slope, knots, weights)
            for course in courses
        ]

    training_courses = tuple(FRESH_COURSES[index] for index in TRAIN_INDICES)
    candidates: list[dict[str, Any]] = []
    generations = []
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        baseline = list(pool.map(_run_task, tasks(FRESH_COURSES, zero)))
        if any(
            row["physical_trace_hash"] != old["physical_trace_hash"]
            for row, old in zip(baseline, router["fresh_candidate"], strict=True)
        ):
            raise ValueError("zero post-contact residual must exactly replay v207")
        mean = np.zeros(12)
        std = np.full(12, 0.22)
        for generation in range(2):
            sampled = np.clip(rng.normal(mean, std, (12, 12)), -0.8, 0.8)
            jobs = [
                job
                for vector in sampled
                for job in tasks(training_courses, tuple(float(value) for value in vector))
            ]
            rollout = list(pool.map(_run_task, jobs))
            ranked = []
            for index, vector in enumerate(sampled):
                weights = tuple(float(value) for value in vector)
                rows = rollout[index * 4 : (index + 1) * 4]
                rank = _rank(rows, weights)
                item = {
                    "generation": generation + 1,
                    "candidate": index,
                    "weights": weights,
                    "training_rows": rows,
                    "rank": rank,
                }
                ranked.append(item)
                candidates.append(item)
            ranked.sort(key=lambda row: row["rank"], reverse=True)
            elite = np.asarray([row["weights"] for row in ranked[:4]], dtype=np.float64)
            mean = elite.mean(axis=0)
            std = np.maximum(0.07, elite.std(axis=0))
            generations.append({"generation": generation + 1, "best_rank": ranked[0]["rank"]})
            print(json.dumps(generations[-1]), flush=True)
        candidates.sort(key=lambda row: row["rank"], reverse=True)
        finalists = []
        for item in candidates[:3]:
            rows = list(pool.map(_run_task, tasks(FRESH_COURSES, tuple(item["weights"]))))
            anchor_rows = list(
                pool.map(_run_task, tasks(TRAIN_COURSES[-2:], tuple(item["weights"])))
            )
            anchor_equal = all(
                row["selected_expert"] == "parent"
                and row["physical_trace_hash"] == old["physical_trace_hash"]
                for row, old in zip(anchor_rows, router["development"][-2:], strict=True)
            )
            finalists.append(
                {
                    "generation": item["generation"],
                    "candidate": item["candidate"],
                    "weights": item["weights"],
                    "training_rank": item["rank"],
                    "development_rows": rows,
                    "development_score": _score(rows),
                    "anchor_rows": anchor_rows,
                    "anchor_physical_equal": anchor_equal,
                }
            )
            print(json.dumps({"finalist": item["candidate"], "score": _score(rows)}), flush=True)
    qualified = [
        row
        for row in finalists
        if row["development_score"][0] >= 3
        and row["development_score"][1] >= 7
        and row["anchor_physical_equal"]
        and all(
            item["safe"] and item["peak_residual_rad"] <= 0.250001
            for item in row["development_rows"]
        )
    ]
    report = {
        "schema": SCHEMA,
        "router_report_hash": router["report_hash"],
        "v212_report_hash": v212["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_V207_FRESH_ONLINE_POSTCONTACT_TRAINING_NOT_NEW_EXAM",
        "seed": SEED,
        "training_indices": TRAIN_INDICES,
        "training_course_seeds": [course.seed for course in training_courses],
        "zero_physical_equal_count": 8,
        "baseline_score": _score(baseline),
        "generations": generations,
        "candidate_count": len(candidates),
        "finalists": finalists,
        "qualified_candidate_count": len(qualified),
        "status": "DEVELOPMENT_POST_CONTACT_CANDIDATE_ONLY"
        if qualified
        else "REJECTED_POST_CONTACT_CEM_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during post-contact CEM")
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
        "router-report",
        "v212-report",
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
        args.router_report,
        args.v212_report,
        args.output,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "baseline_score", "qualified_candidate_count", "report_hash")
            }
        )
    )


if __name__ == "__main__":
    main()
