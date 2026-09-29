"""Frozen SIM_ONLY measured-state receiving composition and new paired exam."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_bounded_middle_fresh_v186 import EXAM_COURSES as CONSUMED_COURSES
from rsi_r1_compliant_piecewise_v174 import run as run_parent
from rsi_r1_side_navigation_fresh_v160 import FRESH_COURSES, clean
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.rsi.receiving_three_zone_expert import ReceivingThreeZoneExpert
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

SCHEMA = "rosclaw_soccer.rsi.r1_three_zone_fresh_v189.result.v1"
# New physical positions and launch speeds; freeze before any v189 execution.
FRESH_COURSES_V189 = (
    ReceivingCourse("red.finisher", 189001, 1.25, 0.066),
    ReceivingCourse("red.finisher", 189002, 1.25, 0.069),
    ReceivingCourse("red.finisher", 189003, 1.25, 0.072),
    ReceivingCourse("red.finisher", 189004, 1.25, 0.074),
    ReceivingCourse("red.finisher", 189005, 1.225, 0.070),
    ReceivingCourse("red.finisher", 189006, 1.275, 0.070),
)


def run_candidate(
    asset_root: Path,
    policy: Path,
    course: ReceivingCourse,
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    slope: tuple[float, ...],
    low: tuple[float, ...],
    center: tuple[float, ...],
    high: tuple[float, ...],
) -> dict[str, Any]:
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingThreeZoneExpert(
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
        low_weights=low,
        center_weights=center,
        high_weights=high,
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
        "selected_zone": feedback.selected_zone,
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


def _find_candidate(report: dict[str, Any], generation: int, index: int) -> dict[str, Any]:
    matches = [
        row
        for group in report["generations"]
        for row in group["rows"]
        if row["generation"] == generation and row["candidate"] == index
    ]
    if len(matches) != 1:
        raise ValueError("sealed development expert missing")
    return matches[0]


def _run_pair(args: tuple[Any, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    asset_root, policy, course, coordination, left, right, slope, low, center, high = args
    candidate = run_candidate(
        asset_root, policy, course, coordination, left, right, slope, low, center, high
    )
    parent = run_parent(asset_root, policy, course, coordination, left, right, slope, (0.0,) * 12)
    return candidate, parent


def exam(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    v187_report: Path,
    v188_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY three-zone exam evidence required")
    fresh_bank_hash = preflight_receiving_courses(FRESH_COURSES_V189)
    parent, right_parent, refine, lateral, v187, v188 = (
        json.loads(path.read_text())
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            v187_report,
            v188_report,
        )
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, refine, lateral, v187, v188)
    ) or (
        v187["status"] != "REJECTED_MIDDLE_TRAINING_GATE"
        or v188["status"] != "REJECTED_SAFE_MIDDLE_REFINEMENT_GATE"
        or v188["previous_report_hash"] != v187["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed rejected development lineage required")
    low_row = _find_candidate(v188, 1, 6)
    center_row = _find_candidate(v188, 2, 7)
    high_row = _find_candidate(v187, 2, 2)
    if not (
        clean(low_row["summaries"][0])
        and low_row["summaries"][0]["controlled_reception"]
        and all(
            clean(center_row["summaries"][index])
            and center_row["summaries"][index]["controlled_reception"]
            for index in (2, 3)
        )
        and clean(high_row["summaries"][1])
        and high_row["summaries"][1]["controlled_reception"]
    ):
        raise ValueError("one expert per observed training zone required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    slope = tuple(lateral["best"]["slope"])
    low = tuple(low_row["middle_weights"])
    center = tuple(center_row["middle_weights"])
    high = tuple(high_row["middle_weights"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_three_zone_fresh_v189.py",
            "src/rosclaw_soccer/rsi/receiving_three_zone_expert.py",
            "src/rosclaw_soccer/rsi/receiving_middle_basis_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    development = [
        run_candidate(
            asset_root, policy, course, coordination, left, right, slope, low, center, high
        )
        for course in CONSUMED_COURSES
    ]
    retention = [
        run_candidate(
            asset_root, policy, course, coordination, left, right, slope, low, center, high
        )
        for course in (FRESH_COURSES[0], COURSES[0])
    ]
    development_pass = all(clean(row) and row["controlled_reception"] for row in development)
    retention_pass = all(clean(row) and row["controlled_reception"] for row in retention)
    candidate_rows: list[dict[str, Any]] = []
    parent_rows: list[dict[str, Any]] = []
    if development_pass and retention_pass:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
            tasks = [
                (asset_root, policy, course, coordination, left, right, slope, low, center, high)
                for course in FRESH_COURSES_V189
            ]
            for index, (candidate, old) in enumerate(pool.map(_run_pair, tasks)):
                candidate_rows.append(candidate)
                parent_rows.append(old)
                (output / "progress.json").write_text(
                    json.dumps({"candidate": candidate_rows, "parent": parent_rows}, indent=2)
                    + "\n"
                )
                print(
                    json.dumps(
                        {
                            "case": index,
                            "zone": candidate["selected_zone"],
                            "candidate": clean(candidate) and candidate["controlled_reception"],
                            "parent": clean(old) and old["controlled_reception"],
                            "nonfoot": candidate["own_nonfoot_frames"],
                            "distance": candidate["tail_maximum_foot_distance_m"],
                            "speed": candidate["tail_maximum_ball_speed_mps"],
                        }
                    ),
                    flush=True,
                )
    candidate_pass = sum(clean(row) and row["controlled_reception"] for row in candidate_rows)
    parent_pass = sum(clean(row) and row["controlled_reception"] for row in parent_rows)
    qualified = (
        development_pass
        and retention_pass
        and candidate_pass == len(FRESH_COURSES_V189)
        and candidate_pass > parent_pass
    )
    report = {
        "schema": SCHEMA,
        "v187_report_hash": v187["report_hash"],
        "v188_report_hash": v188["report_hash"],
        "source_hashes": sources,
        "partition": "FROZEN_NEW_THREE_ZONE_RECEIVING_HOLDOUT",
        "fresh_bank_hash": fresh_bank_hash,
        "fresh_courses": [vars(course) for course in FRESH_COURSES_V189],
        "selected_experts": {"low": [1, 6], "center": [2, 7], "high": [2, 2]},
        "development": development,
        "development_pass": development_pass,
        "retention": retention,
        "retention_pass": retention_pass,
        "candidate": candidate_rows,
        "parent": parent_rows,
        "candidate_pass": candidate_pass,
        "parent_pass": parent_pass,
        "status": "QUALIFIED_LOCAL_RECEIVING_BASIN_ONLY"
        if qualified
        else "REJECTED_THREE_ZONE_FRESH_GATE",
        "promotion_authorized": False,
        "team_video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during three-zone frozen exam")
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = exam(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.v187_report,
        args.v188_report,
        args.output,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
