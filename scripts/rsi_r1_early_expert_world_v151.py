"""SIM_ONLY eight-G1 early-admission A2 expert feasibility sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_precontact_expert_world_v150 import PHYSICAL_KEYS
from rsi_r1_precontact_expert_world_v150 import run as parent_run
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.rsi.receiving_early_precontact_expert import ReceivingEarlyPrecontactExpert
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_early_expert_world_v151.result.v1"
ENTRIES = (7, 10, 12)


def run_early(
    asset_root: Path,
    policy: Path,
    course: ReceivingCourse,
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    entry: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", entry, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    actor = ReceivingEarlyPrecontactExpert(
        course.agent_id,
        schedule.contract_hash,
        mailbox,
        0.35,
        0.0,
        0.0,
        target_depth_m=0.25,
        target_lateral_m=0.12,
        coordination=coordination,
        left_weights=left,
        right_weights=right,
        entry_frame=entry,
        full_frame=18,
    )
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy,
        course=course,
        scenario_id=f"s199.rsi.r1.coordinated-receiving.{course.seed}",
        configuration_profile="R1_CONTACT_TAP",
        oracle=schedule,
        feedback_provider=actor,
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
    force = np.asarray(trace["ball_contact_force_n"])
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    nonfoot_force = np.asarray(trace["ball_nonfoot_contact_force_n"])
    first = next(
        (frame for frame in range(20, 120) if foot[frame] == code and force[frame] > 0),
        None,
    )
    nonfoot_frames = [
        frame for frame in range(20, 120) if nonfoot[frame] == code and nonfoot_force[frame] > 0
    ]
    summary = {
        "course": vars(course),
        "entry_frame": entry,
        "safe": info["safe"],
        "fault_agents": info["physics_evidence_fault_agents"],
        "first_foot_frame": first,
        "own_nonfoot_frames": nonfoot_frames,
        "controlled_reception": outcome["controlled_reception"],
        "tail_maximum_foot_distance_m": detail["tail_maximum_foot_distance_m"],
        "tail_maximum_ball_speed_mps": detail["tail_maximum_ball_speed_mps"],
        "peak_actual_residual_rad": float(
            np.max(np.abs(np.asarray(trace["receiving_oracle_delta_rad"])))
        ),
        "residual_at_frame_18_rad": float(
            np.max(np.abs(np.asarray(trace["receiving_oracle_delta_rad"])[18]))
        ),
        "result_hash": hash_json(info),
    }
    return summary, trace


def sweep(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    experts_report: Path,
    right_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY eight-G1 evidence required")
    parent = json.loads(parent_report.read_text())
    experts = json.loads(experts_report.read_text())
    right_parent = json.loads(right_report.read_text())
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, experts, right_parent)
    ) or (
        experts["parent_report_hash"] != right_parent["report_hash"]
        or experts["status"] != "DEVELOPMENT_BILATERAL_CONTROLLED_UNVALIDATED"
        or parent["selected"] is None
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed parent and proxy expert reports required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(experts["best_left"]["weights"])
    right = tuple(right_parent["best_right"]["weights"])
    source_hashes = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_early_expert_world_v151.py",
            "src/rosclaw_soccer/rsi/receiving_early_precontact_expert.py",
            "src/rosclaw_soccer/training/receiving_oracle_schedule.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    zero_rows = []
    for course in COURSES:
        baseline, baseline_trace = parent_run(asset_root, policy, course, coordination, None, None)
        zero, zero_trace = run_early(
            asset_root,
            policy,
            course,
            coordination,
            (0.0,) * 12,
            (0.0,) * 12,
            7,
        )
        equal = all(
            np.array_equal(np.asarray(baseline_trace[key]), np.asarray(zero_trace[key]))
            for key in PHYSICAL_KEYS
        )
        zero_rows.append(
            {"course": vars(course), "equal": equal, "baseline": baseline, "zero": zero}
        )
        if not equal:
            break
    rows = []
    if len(zero_rows) == 2 and all(item["equal"] for item in zero_rows):
        for entry in ENTRIES:
            for course in COURSES:
                summary, _ = run_early(asset_root, policy, course, coordination, left, right, entry)
                rows.append(summary)
            (output / "progress.json").write_text(
                json.dumps(rows, indent=2, allow_nan=False) + "\n"
            )
            print(json.dumps({"entry": entry, "rows": rows[-2:]}), flush=True)
    zero_qualified = len(zero_rows) == 2 and all(item["equal"] for item in zero_rows)
    paired_controlled = (
        any(
            all(
                row["safe"]
                and not row["fault_agents"]
                and row["first_foot_frame"] is not None
                and not row["own_nonfoot_frames"]
                and row["controlled_reception"]
                for row in rows[index : index + 2]
            )
            for index in (0, 2, 4)
        )
        if len(rows) == 6
        else False
    )
    report = {
        "schema": SCHEMA,
        "parent_report_hash": parent["report_hash"],
        "experts_report_hash": experts["report_hash"],
        "source_hashes": source_hashes,
        "zero_rows": zero_rows,
        "candidate_rows": rows,
        "status": "DEVELOPMENT_EIGHT_G1_PAIRED_CONTROLLED_UNVALIDATED"
        if paired_controlled
        else "REJECTED_EIGHT_G1_PAIRED_GATE"
        if zero_qualified
        else "REJECTED_ZERO_AUTHORITY_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(
        hash_bytes((root / name).read_bytes()) != digest for name, digest in source_hashes.items()
    ):
        raise ValueError("source drift during early A2 eight-G1 experiment")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--experts-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = sweep(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.experts_report,
        args.right_report,
        args.output,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
