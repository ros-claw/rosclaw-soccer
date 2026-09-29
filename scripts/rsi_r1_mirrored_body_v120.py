"""Consumed-course left/right mirrored A1 receiving development, SIM_ONLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.receiving_mirrored_body_residual import ReceivingMirroredBodyResidual
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

COURSES = (
    ReceivingCourse("red.finisher", 92801, 1.25, 0.08),
    ReceivingCourse("red.finisher", 92803, 1.5, -0.08),
)
SCHEDULE = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))


def evaluate(
    asset_root: Path, policy: Path, course: ReceivingCourse, coefficients: tuple[float, ...]
) -> dict[str, Any]:
    mailbox = ReceiveContactMailbox(course.agent_id)
    actor = ReceivingMirroredBodyResidual(
        course.agent_id, SCHEDULE.contract_hash, mailbox, coefficients
    )
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy,
        course=course,
        scenario_id=f"s199.rsi.r1.mirrored-body.{course.seed}",
        configuration_profile="R1_CONTACT_TAP",
        oracle=SCHEDULE,
        feedback_provider=actor,
        physics_evidence_consumers={course.agent_id: ReceiveContactEvidence(mailbox)},
    )
    result_dict = result.to_dict()
    ids = tuple(sorted(row["agent_id"] for row in result_dict["qualities"]))
    _, outcome = receiving_window(
        trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
    )
    detail = explain_receiving_window(
        trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
    )
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])[20:120]
    nonfoot_force = np.asarray(trace["ball_nonfoot_contact_force_n"])[20:120]
    first = outcome["first_foot_contact_sec"]
    frames = np.flatnonzero((nonfoot > 0) & (nonfoot_force > 0)) + 20
    events = [
        {
            "frame": int(frame),
            "time_sec": float(np.asarray(trace["time"])[frame]),
            "agent_code": int(np.asarray(trace["ball_nonfoot_contact_agent_code"])[frame]),
            "geom_id": int(np.asarray(trace["ball_nonfoot_contact_geom_id"])[frame]),
            "force_n": float(np.asarray(trace["ball_nonfoot_contact_force_n"])[frame]),
        }
        for frame in frames
        if first is not None and float(np.asarray(trace["time"])[frame]) >= first
    ]
    return {
        "course": vars(course),
        "safe": result_dict["safe"],
        "physics_evidence_fault_agents": result_dict["physics_evidence_fault_agents"],
        "result_hash": hash_json(result_dict),
        "control_frames": len(trace["time"]),
        "nonzero_proposal_frames": actor.nonzero_frames,
        "first_own_foot_time_sec": mailbox.snapshot.first_own_foot_time_sec,
        "first_own_foot": mailbox.snapshot.first_own_foot,
        "prefoot_nonfoot_count": mailbox.snapshot.prefoot_nonfoot_count,
        "outcome": outcome,
        "explanation": detail,
        "postfoot_nonfoot_events": events,
    }


def train(asset_root: Path, policy: Path, parent_report: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("new external evidence directory required")
    parent = json.loads(parent_report.read_text())
    if (
        parent["schema"] != "rosclaw_soccer.rsi.r1_whole_body_residual_v119.result.v1"
        or parent["report_hash"]
        != hash_json({key: value for key, value in parent.items() if key != "report_hash"})
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
        or parent["selected"]["controlled_count"] != 0
    ):
        raise ValueError("frozen failed v119 development foundation required")
    root = Path(__file__).resolve().parents[1]
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_mirrored_body_v120.py",
            "src/rosclaw_soccer/rsi/receiving_mirrored_body_residual.py",
            "src/rosclaw_soccer/rsi/receiving_whole_body_residual.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/training/receiving_rollout.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    rng = np.random.default_rng(1200929)
    c3 = np.asarray(parent["all_candidates"][3]["coefficients"], dtype=np.float64)
    c6 = np.asarray(parent["all_candidates"][6]["coefficients"], dtype=np.float64)
    vectors = [np.zeros(12), c3, c6]
    vectors += [np.clip(c3 + rng.normal(0.0, 0.2, 12), -0.8, 0.8) for _ in range(8)]
    output.mkdir(parents=True)
    rows = []
    for i, vector in enumerate(vectors):
        coefficients = tuple(float(x) for x in vector)
        trials = [evaluate(asset_root, policy, course, coefficients) for course in COURSES]
        safe = all(t["safe"] and not t["physics_evidence_fault_agents"] for t in trials)
        foot = all(
            t["first_own_foot_time_sec"] is not None and t["prefoot_nonfoot_count"] == 0
            for t in trials
        )
        row = {
            "candidate": i,
            "coefficients": coefficients,
            "safe": safe,
            "all_clean_first_foot": foot,
            "controlled_count": sum(t["outcome"]["controlled_reception"] for t in trials),
            "postfoot_nonfoot_frame_count": sum(len(t["postfoot_nonfoot_events"]) for t in trials),
            "mean_shaped_return": float(np.mean([t["outcome"]["shaped_return"] for t in trials])),
            "trials": trials,
        }
        rows.append(row)
        (output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "candidate": i,
                    "safe": safe,
                    "controlled": row["controlled_count"],
                    "nonfoot": row["postfoot_nonfoot_frame_count"],
                    "failed": [t["explanation"]["failed_criteria"] for t in trials],
                }
            ),
            flush=True,
        )
    eligible = [row for row in rows if row["safe"] and row["all_clean_first_foot"]]
    if not eligible:
        raise ValueError("no safe clean-first-foot candidate; all development evidence retained")
    best = max(
        eligible,
        key=lambda row: (
            row["controlled_count"],
            -row["postfoot_nonfoot_frame_count"],
            row["mean_shaped_return"],
        ),
    )
    report = {
        "schema": "rosclaw_soccer.rsi.r1_mirrored_body_v120.result.v1",
        "partition": "CONSUMED_EIGHT_G1_DEVELOPMENT",
        "source_hashes": sources,
        "v119_report_hash": parent["report_hash"],
        "policy_hash": hash_bytes(policy.read_bytes()),
        "schedule_hash": SCHEDULE.contract_hash,
        "candidate_count": len(rows),
        "rollout_count": len(rows) * len(COURSES),
        "parent": rows[0],
        "selected": best,
        "all_candidates": rows,
        "status": "DEVELOPMENT_CONTROLLED_GAIN_UNVALIDATED"
        if best["controlled_count"] > rows[0]["controlled_count"]
        else "REJECTED_NO_CONTROLLED_GAIN",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during development")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train(args.asset_root, args.policy, args.parent_report, args.output)
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
