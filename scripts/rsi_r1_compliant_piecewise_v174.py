"""SIM_ONLY native eight-G1 training of the far segment of a receiving policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_side_navigation_fresh_v160 import FRESH_COURSES, clean
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.rsi.receiving_lateral_piecewise_expert import ReceivingLateralPiecewiseExpert
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

SCHEMA = "rosclaw_soccer.rsi.r1_compliant_piecewise_v174.result.v1"
POPULATION = 12
GENERATIONS = 2


def run(
    asset_root: Path,
    policy: Path,
    course: ReceivingCourse,
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    old_slope: tuple[float, ...],
    far_slope: tuple[float, ...],
) -> dict[str, Any]:
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingLateralPiecewiseExpert(
        course.agent_id,
        schedule.contract_hash,
        mailbox,
        coordination,
        left,
        right,
        left_post_gain=0.0,
        right_post_gain=0.4,
        old_lateral_slope=old_slope,
        far_lateral_slope=far_slope,
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
        reference_policy_path=policy,
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
        "measured_lateral_m": feedback.measured_lateral_m,
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
        "result_hash": hash_json(info),
    }


def rank(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(clean(row) and row["controlled_reception"]),
        float(clean(row)),
        -float(len(row["own_nonfoot_frames"])),
        -float(row["tail_maximum_foot_distance_m"]),
        -float(row["tail_maximum_ball_speed_mps"]),
    )


def train(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    output: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root) or not 0 <= seed < 2**31:
        raise ValueError("new external SIM_ONLY piecewise evidence required")
    parent, right_parent, refine, lateral = (
        json.loads(path.read_text())
        for path in (parent_report, right_report, refine_report, lateral_report)
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, refine, lateral)
    ) or (
        lateral["status"] != "DEVELOPMENT_TWO_COURSE_CONTROLLED_UNVALIDATED"
        or refine["status"] != "DEVELOPMENT_LOCAL_COMPLIANT_CONTROLLED_UNVALIDATED"
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed local and conditioned anchors required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    old_slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_compliant_piecewise_v174.py",
            "src/rosclaw_soccer/rsi/receiving_lateral_piecewise_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    mean = np.zeros(12)
    std = np.full(12, 0.12)
    generations: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for generation in range(GENERATIONS):
        candidates = np.clip(rng.normal(mean, std, size=(POPULATION, 12)), -1.0, 1.0)
        candidates[0] = 0.0 if generation == 0 else mean
        rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            slope = tuple(float(value) for value in candidate)
            summary = run(
                asset_root,
                policy,
                FRESH_COURSES[1],
                coordination,
                left,
                right,
                old_slope,
                slope,
            )
            if summary["active_substeps"] > 32:
                raise ValueError("compliance budget exceeded")
            row = {
                "generation": generation + 1,
                "candidate": index,
                "far_slope": slope,
                "summary": summary,
            }
            rows.append(row)
            if best is None or rank(summary) > rank(best["summary"]):
                best = row
            (output / "progress.json").write_text(
                json.dumps([*generations, {"generation": generation + 1, "rows": rows}], indent=2)
                + "\n"
            )
            print(
                json.dumps(
                    {
                        "generation": generation + 1,
                        "candidate": index,
                        "clean": clean(summary),
                        "controlled": summary["controlled_reception"],
                        "nonfoot": summary["own_nonfoot_frames"],
                        "distance": summary["tail_maximum_foot_distance_m"],
                        "speed": summary["tail_maximum_ball_speed_mps"],
                    }
                ),
                flush=True,
            )
        rows.sort(key=lambda row: rank(row["summary"]), reverse=True)
        elite = np.stack([np.asarray(row["far_slope"]) for row in rows[:4]])
        mean = elite.mean(axis=0)
        std = np.maximum(0.04, elite.std(axis=0))
        generations.append({"generation": generation + 1, "rows": rows})
    assert best is not None
    selected = tuple(best["far_slope"])
    checks = [
        run(asset_root, policy, course, coordination, left, right, old_slope, selected)
        for course in (FRESH_COURSES[0], COURSES[0], FRESH_COURSES[1])
    ]
    positive = all(clean(row) and row["controlled_reception"] for row in checks)
    report = {
        "schema": SCHEMA,
        "lateral_report_hash": lateral["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_PIECEWISE_TRAINING",
        "seed": seed,
        "generations": generations,
        "best": best,
        "checks": checks,
        "status": "DEVELOPMENT_THREE_COURSE_CONTROLLED_UNVALIDATED"
        if positive
        else "REJECTED_THREE_COURSE_CONTROLLED_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during piecewise training")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--refine-report", type=Path, required=True)
    parser.add_argument("--lateral-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=174929)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.output,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
