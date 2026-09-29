"""SIM_ONLY eight-G1 measured-foot precontact-action release sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_precontact_expert_world_v150 import PHYSICAL_KEYS
from rsi_r1_precontact_expert_world_v150 import run as parent_run
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.rsi.receiving_posttouch_release_expert import ReceivingPosttouchReleaseExpert
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_posttouch_release_world_v153.result.v1"
RELEASES = (0, 1, 2, 4)


def run_release(
    asset_root: Path,
    policy: Path,
    course: ReceivingCourse,
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    release: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    actor = ReceivingPosttouchReleaseExpert(
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
        release_frames=release,
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
    summary = {
        "course": vars(course),
        "release_frames": release,
        "safe": info["safe"],
        "fault_agents": info["physics_evidence_fault_agents"],
        "first_foot_frame": first,
        "own_nonfoot_frames": [
            frame for frame in range(20, 120) if nonfoot[frame] == code and nonfoot_force[frame] > 0
        ],
        "controlled_reception": outcome["controlled_reception"],
        "tail_maximum_foot_distance_m": detail["tail_maximum_foot_distance_m"],
        "tail_maximum_ball_speed_mps": detail["tail_maximum_ball_speed_mps"],
        "peak_actual_residual_rad": float(
            np.max(np.abs(np.asarray(trace["receiving_oracle_delta_rad"])))
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
    scale_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY posttouch evidence required")
    parent = json.loads(parent_report.read_text())
    experts = json.loads(experts_report.read_text())
    right_parent = json.loads(right_report.read_text())
    scale = json.loads(scale_report.read_text())
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, experts, right_parent, scale)
    ) or (
        scale["status"] != "REJECTED_NATIVE_BILATERAL_CONTROLLED_GATE"
        or experts["parent_report_hash"] != right_parent["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed native scale and conditional experts required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(float(v) for v in np.asarray(experts["best_left"]["weights"]) * 0.55)
    right = tuple(float(v) for v in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_posttouch_release_world_v153.py",
            "src/rosclaw_soccer/rsi/receiving_posttouch_release_expert.py",
            "src/rosclaw_soccer/training/receiving_oracle_schedule.py",
        )
    }
    output.mkdir(parents=True)
    zero_rows = []
    for course in COURSES:
        baseline, baseline_trace = parent_run(asset_root, policy, course, coordination, None, None)
        zero, zero_trace = run_release(
            asset_root, policy, course, coordination, (0.0,) * 12, (0.0,) * 12, 0
        )
        equal = all(
            np.array_equal(np.asarray(baseline_trace[key]), np.asarray(zero_trace[key]))
            for key in PHYSICAL_KEYS
        )
        zero_rows.append({"course": vars(course), "equal": equal, "zero": zero})
        if not equal:
            break
    rows = []
    if len(zero_rows) == 2 and all(item["equal"] for item in zero_rows):
        for release in RELEASES:
            for course in COURSES:
                summary, _ = run_release(
                    asset_root, policy, course, coordination, left, right, release
                )
                rows.append(summary)
            (output / "progress.json").write_text(
                json.dumps(rows, indent=2, allow_nan=False) + "\n"
            )
            print(json.dumps({"release": release, "rows": rows[-2:]}), flush=True)
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
            for index in (0, 2, 4, 6)
        )
        if len(rows) == 8
        else False
    )
    report = {
        "schema": SCHEMA,
        "scale_report_hash": scale["report_hash"],
        "source_hashes": sources,
        "zero_rows": zero_rows,
        "candidate_rows": rows,
        "status": "DEVELOPMENT_NATIVE_PAIRED_CONTROLLED_UNVALIDATED"
        if paired_controlled
        else "REJECTED_NATIVE_PAIRED_CONTROLLED_GATE"
        if zero_qualified
        else "REJECTED_ZERO_AUTHORITY_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during native posttouch release search")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--experts-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--scale-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = sweep(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.experts_report,
        args.right_report,
        args.scale_report,
        args.output,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
