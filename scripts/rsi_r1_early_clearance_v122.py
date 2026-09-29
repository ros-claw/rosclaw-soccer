"""Paired 8-G1 search of early geometry-informed foot/shin clearance, SIM_ONLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.receiving_shin_clearance_residual import ReceivingShinClearanceResidual
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
SCHEDULE = ReceivingOracleSchedule("red.finisher", "A1_body29", 0, 10, ((0.0,) * 29,))
PARAMETERS = (
    (0.0, 0.0, 0.0),
    (0.03, 0.0, -0.05),
    (0.05, 0.0, -0.05),
    (0.05, 0.0, -0.1),
    (0.08, 0.0, -0.05),
    (0.08, 0.0, -0.1),
    (0.05, 0.05, -0.1),
    (0.08, 0.05, -0.1),
)


def evaluate(
    asset_root: Path, policy: Path, course: ReceivingCourse, parameters: tuple[float, float, float]
) -> dict[str, Any]:
    mailbox = ReceiveContactMailbox(course.agent_id)
    actor = ReceivingShinClearanceResidual(
        course.agent_id,
        SCHEDULE.contract_hash,
        mailbox,
        (0.0,) * 12,
        *parameters,
    )
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy,
        course=course,
        scenario_id=f"s199.rsi.r1.early-clearance.{course.seed}",
        configuration_profile="R1_CONTACT_TAP",
        oracle=SCHEDULE,
        feedback_provider=actor,
        physics_evidence_consumers={course.agent_id: ReceiveContactEvidence(mailbox)},
    )
    info = result.to_dict()
    ids = tuple(sorted(row["agent_id"] for row in info["qualities"]))
    _, outcome = receiving_window(
        trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
    )
    explanation = explain_receiving_window(
        trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
    )
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    force = np.asarray(trace["ball_nonfoot_contact_force_n"])
    contact = np.asarray(trace["ball_contact_agent_code"])
    effector = np.asarray(trace["ball_contact_effector_code"])
    contact_force = np.asarray(trace["ball_contact_force_n"])
    first_frame = next(
        (
            i
            for i in range(20, 120)
            if contact[i] == ids.index(course.agent_id) + 1
            and effector[i] in (1, 2)
            and contact_force[i] > 0
        ),
        None,
    )
    event_frames = (
        []
        if first_frame is None
        else [
            i
            for i in range(first_frame, 120)
            if nonfoot[i] == ids.index(course.agent_id) + 1 and force[i] > 0
        ]
    )
    return {
        "course": vars(course),
        "safe": info["safe"],
        "physics_evidence_fault_agents": info["physics_evidence_fault_agents"],
        "result_hash": hash_json(info),
        "control_frames": len(trace["time"]),
        "nonzero_proposal_frames": actor.nonzero_frames,
        "first_own_foot_time_sec": mailbox.snapshot.first_own_foot_time_sec,
        "first_own_foot": mailbox.snapshot.first_own_foot,
        "prefoot_nonfoot_count": mailbox.snapshot.prefoot_nonfoot_count,
        "postfoot_own_shin_events": [
            {
                "frame": frame,
                "geom_id": int(np.asarray(trace["ball_nonfoot_contact_geom_id"])[frame]),
                "force_n": float(force[frame]),
            }
            for frame in event_frames
        ],
        "outcome": outcome,
        "explanation": explanation,
    }


def train(asset_root: Path, policy: Path, geometry_report: Path, output: Path) -> dict[str, Any]:
    if output.exists() or output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("new external evidence directory required")
    geometry = json.loads(geometry_report.read_text())
    if (
        geometry["schema"] != "rosclaw_soccer.rsi.r1_shin_clearance_v121.result.v1"
        or geometry["report_hash"]
        != hash_json({key: value for key, value in geometry.items() if key != "report_hash"})
        or geometry["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("frozen measured geometry foundation required")
    root = Path(__file__).resolve().parents[1]
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_early_clearance_v122.py",
            "src/rosclaw_soccer/rsi/receiving_shin_clearance_residual.py",
            "src/rosclaw_soccer/rsi/receiving_whole_body_residual.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/training/receiving_rollout.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rows = []
    for i, parameters in enumerate(PARAMETERS):
        trials = [evaluate(asset_root, policy, course, parameters) for course in COURSES]
        row = {
            "candidate": i,
            "parameters_rad": parameters,
            "safe": all(t["safe"] and not t["physics_evidence_fault_agents"] for t in trials),
            "all_clean_first_foot": all(
                t["first_own_foot_time_sec"] is not None and t["prefoot_nonfoot_count"] == 0
                for t in trials
            ),
            "controlled_count": sum(t["outcome"]["controlled_reception"] for t in trials),
            "own_shin_event_count": sum(len(t["postfoot_own_shin_events"]) for t in trials),
            "mean_shaped_return": float(np.mean([t["outcome"]["shaped_return"] for t in trials])),
            "trials": trials,
        }
        rows.append(row)
        (output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "candidate": i,
                    "safe": row["safe"],
                    "foot": row["all_clean_first_foot"],
                    "controlled": row["controlled_count"],
                    "own_shin": row["own_shin_event_count"],
                    "failed": [t["explanation"]["failed_criteria"] for t in trials],
                }
            ),
            flush=True,
        )
    eligible = [row for row in rows if row["safe"] and row["all_clean_first_foot"]]
    if not eligible:
        raise ValueError("no safe clean-first-foot candidate; progress evidence retained")
    best = max(
        eligible,
        key=lambda row: (
            row["controlled_count"],
            -row["own_shin_event_count"],
            row["mean_shaped_return"],
        ),
    )
    report = {
        "schema": "rosclaw_soccer.rsi.r1_early_clearance_v122.result.v1",
        "partition": "CONSUMED_EIGHT_G1_DEVELOPMENT",
        "source_hashes": sources,
        "geometry_report_hash": geometry["report_hash"],
        "policy_hash": hash_bytes(policy.read_bytes()),
        "schedule_hash": SCHEDULE.contract_hash,
        "candidate_count": len(rows),
        "rollout_count": len(rows) * len(COURSES),
        "parent": rows[0],
        "selected": best,
        "all_candidates": rows,
        "status": "DEVELOPMENT_CONTROLLED_GAIN_UNVALIDATED"
        if best["controlled_count"] > rows[0]["controlled_count"]
        else "DEVELOPMENT_SHIN_GAIN_UNVALIDATED"
        if best["own_shin_event_count"] < rows[0]["own_shin_event_count"]
        else "REJECTED_NO_PHYSICAL_GAIN",
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
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train(args.asset_root, args.policy, args.geometry_report, args.output)
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
