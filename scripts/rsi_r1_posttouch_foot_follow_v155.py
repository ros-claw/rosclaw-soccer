"""SIM_ONLY native eight-G1 post-foot task-space follow on clean-contact experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.rsi.receiving_precontact_expert import ReceivingPrecontactExpert
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_posttouch_foot_follow_v155.result.v1"
GAINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def run_gain(
    asset_root: Path,
    policy: Path,
    course: ReceivingCourse,
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    gain: float,
) -> dict[str, Any]:
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    actor = ReceivingPrecontactExpert(
        course.agent_id,
        schedule.contract_hash,
        mailbox,
        0.35,
        gain,
        0.0,
        target_depth_m=0.25,
        target_lateral_m=0.12,
        coordination=coordination,
        left_weights=left,
        right_weights=right,
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
    foot_force = np.asarray(trace["ball_contact_force_n"])
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    nonfoot_force = np.asarray(trace["ball_nonfoot_contact_force_n"])
    return {
        "course": vars(course),
        "post_gain": gain,
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
        "peak_actual_residual_rad": float(
            np.max(np.abs(np.asarray(trace["receiving_oracle_delta_rad"])))
        ),
        "result_hash": hash_json(info),
    }


def sweep(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    native_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY post-foot evidence required")
    parent = json.loads(parent_report.read_text())
    right_parent = json.loads(right_report.read_text())
    native = json.loads(native_report.read_text())
    if any(
        row["report_hash"] != hash_json({k: v for k, v in row.items() if k != "report_hash"})
        for row in (parent, right_parent, native)
    ) or (
        native["status"] != "REJECTED_NATIVE_PAIRED_CONTROLLED_GATE"
        or native["paired"] is None
        or not all(
            row["safe"] and row["first_foot_frame"] is not None and not row["own_nonfoot_frames"]
            for row in native["paired"]
        )
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed native bilateral clean-contact experts required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(native["best"]["weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_posttouch_foot_follow_v155.py",
            "src/rosclaw_soccer/rsi/receiving_precontact_expert.py",
            "src/rosclaw_soccer/training/receiving_oracle_schedule.py",
        )
    }
    output.mkdir(parents=True)
    rows = []
    for gain in GAINS:
        for course in COURSES:
            summary = run_gain(asset_root, policy, course, coordination, left, right, gain)
            rows.append(summary)
        (output / "progress.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"gain": gain, "rows": rows[-2:]}), flush=True)
    paired_controlled = any(
        all(
            row["safe"]
            and not row["fault_agents"]
            and row["first_foot_frame"] is not None
            and not row["own_nonfoot_frames"]
            and row["controlled_reception"]
            for row in rows[index : index + 2]
        )
        for index in range(0, len(rows), 2)
    )
    report = {
        "schema": SCHEMA,
        "native_report_hash": native["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_POST_FOOT_TASKSPACE_FOLLOW",
        "rollout_count": len(rows),
        "rows": rows,
        "status": "DEVELOPMENT_NATIVE_PAIRED_CONTROLLED_UNVALIDATED"
        if paired_controlled
        else "REJECTED_NATIVE_PAIRED_CONTROLLED_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during post-foot feedback learning")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--native-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = sweep(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.native_report,
        args.output,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
