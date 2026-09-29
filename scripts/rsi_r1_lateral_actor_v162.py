"""SIM_ONLY state-conditioned pre-contact adaptation over consumed left ball positions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_posttouch_navigation_v158 import POST_WEIGHTS
from rsi_r1_side_navigation_fresh_v160 import FRESH_COURSES, clean
from rsi_r1_side_navigation_fresh_v160 import run_candidate as run_parent
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.rsi.receiving_lateral_conditioned_expert import (
    ReceivingLateralConditionedExpert,
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

SCHEMA = "rosclaw_soccer.rsi.r1_lateral_actor_v162.result.v1"
POPULATION = 8
GENERATIONS = 2
TARGET_COURSES = (FRESH_COURSES[0], FRESH_COURSES[1])


def run(
    asset_root: Path,
    policy: Path,
    course: ReceivingCourse,
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    slope: tuple[float, ...],
) -> dict[str, Any]:
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingLateralConditionedExpert(
        course.agent_id,
        schedule.contract_hash,
        mailbox,
        coordination,
        left,
        right,
        left_post_gain=0.0,
        right_post_gain=0.4,
        left_lateral_slope=slope,
    )
    navigation = TeamReceiveSideNavigation(
        course.agent_id,
        hash_bytes((asset_root / "policy/loco_mode/model/policy_29dof.pt").read_bytes()),
        hash_bytes((asset_root / "policy/loco_mode/config/LocoMode.yaml").read_bytes()),
        mailbox,
        POST_WEIGHTS[6],
        POST_WEIGHTS[0],
    )
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy,
        course=course,
        scenario_id=f"s199.rsi.r1.coordinated-receiving.{course.seed}",
        configuration_profile="R1_CONTACT_TAP",
        oracle=schedule,
        feedback_provider=feedback,
        research_navigation_policy=navigation,
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
    return {
        "course": vars(course),
        "measured_lateral_m": feedback.measured_lateral_m,
        "selected_side": feedback._selected_side,
        "navigation_selected_side": navigation.selected_side,
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
        "result_hash": hash_json(info),
    }


def rank(rows: list[dict[str, Any]]) -> tuple[float, ...]:
    return (
        float(sum(clean(row) and row["controlled_reception"] for row in rows)),
        float(sum(clean(row) for row in rows)),
        -float(sum(len(row["own_nonfoot_frames"]) for row in rows)),
        -float(max(row["tail_maximum_foot_distance_m"] for row in rows)),
        -float(max(row["tail_maximum_ball_speed_mps"] for row in rows)),
    )


def train(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    joint_report: Path,
    multicourse_report: Path,
    output: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root) or not 0 <= seed < 2**31:
        raise ValueError("new external SIM_ONLY lateral-actor evidence required")
    parent, right_parent, joint, multicourse = (
        json.loads(path.read_text())
        for path in (parent_report, right_report, joint_report, multicourse_report)
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, joint, multicourse)
    ) or (
        multicourse["status"] != "REJECTED_MULTICOURSE_CONTROLLED_GATE"
        or joint["status"] != "DEVELOPMENT_NATIVE_BILATERAL_CONTROLLED_UNVALIDATED"
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed consumed multi-course lineage required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(joint["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_lateral_actor_v162.py",
            "src/rosclaw_soccer/rsi/receiving_lateral_conditioned_expert.py",
            "src/rosclaw_soccer/rsi/team_receive_side_navigation.py",
        )
    }
    output.mkdir(parents=True)
    baseline = []
    baseline_equivalent = True
    for course in (COURSES[0], *TARGET_COURSES, COURSES[1]):
        zero = run(asset_root, policy, course, coordination, left, right, (0.0,) * 12)
        old = run_parent(asset_root, policy, course, coordination, left, right)
        keys = (
            "safe",
            "fault_agents",
            "first_foot_frame",
            "own_nonfoot_frames",
            "controlled_reception",
            "tail_maximum_foot_distance_m",
            "tail_maximum_ball_speed_mps",
        )
        equal = all(zero[key] == old[key] for key in keys)
        baseline.append({"course": vars(course), "zero": zero, "old": old, "equal": equal})
        baseline_equivalent &= equal
        print(
            json.dumps(
                {
                    "partition": "zero",
                    "course": vars(course),
                    "equal": equal,
                    "measured_lateral_m": zero["measured_lateral_m"],
                }
            ),
            flush=True,
        )
    rng = np.random.default_rng(seed)
    mean = np.zeros(12)
    std = np.full(12, 0.12)
    generations: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    if baseline_equivalent:
        for generation in range(GENERATIONS):
            candidates = np.clip(rng.normal(mean, std, size=(POPULATION, 12)), -1.0, 1.0)
            candidates[0] = np.zeros(12) if generation == 0 else mean
            rows: list[dict[str, Any]] = []
            for index, candidate in enumerate(candidates):
                slope = tuple(float(value) for value in candidate)
                summaries = [
                    run(asset_root, policy, course, coordination, left, right, slope)
                    for course in TARGET_COURSES
                ]
                row = {
                    "generation": generation + 1,
                    "candidate": index,
                    "slope": slope,
                    "summaries": summaries,
                }
                rows.append(row)
                if best is None or rank(summaries) > rank(best["summaries"]):
                    best = row
                (output / "progress.json").write_text(
                    json.dumps(
                        [*generations, {"generation": generation + 1, "rows": rows}],
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                )
                print(
                    json.dumps(
                        {
                            "generation": generation + 1,
                            "candidate": index,
                            "clean": [clean(item) for item in summaries],
                            "controlled": [item["controlled_reception"] for item in summaries],
                            "nonfoot": [item["own_nonfoot_frames"] for item in summaries],
                            "distance": [
                                item["tail_maximum_foot_distance_m"] for item in summaries
                            ],
                        }
                    ),
                    flush=True,
                )
            rows.sort(key=lambda item: rank(item["summaries"]), reverse=True)
            elite = np.stack([np.asarray(row["slope"]) for row in rows[:3]])
            mean = elite.mean(axis=0)
            std = np.maximum(0.035, elite.std(axis=0))
            generations.append({"generation": generation + 1, "rows": rows})
    checks = []
    if best is not None:
        checks = [
            run(asset_root, policy, course, coordination, left, right, tuple(best["slope"]))
            for course in (COURSES[0], *TARGET_COURSES, COURSES[1])
        ]
    all_controlled = len(checks) == 4 and all(
        clean(row) and row["controlled_reception"] for row in checks
    )
    report = {
        "schema": SCHEMA,
        "multicourse_report_hash": multicourse["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_MEASURED_LATERAL_CURRICULUM",
        "seed": seed,
        "baseline_equivalent": baseline_equivalent,
        "baseline": baseline,
        "generations": generations,
        "best": best,
        "checks": checks,
        "status": "DEVELOPMENT_CONDITIONAL_MULTICOURSE_UNVALIDATED"
        if all_controlled
        else "REJECTED_CONDITIONAL_MULTICOURSE_GATE"
        if baseline_equivalent
        else "REJECTED_ZERO_EQUIVALENCE_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during lateral policy learning")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--joint-report", type=Path, required=True)
    parser.add_argument("--multicourse-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=162928)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.joint_report,
        args.multicourse_report,
        args.output,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
