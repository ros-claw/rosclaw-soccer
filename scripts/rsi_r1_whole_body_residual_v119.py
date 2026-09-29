"""SIM_ONLY 8-G1 paired physical search for a bounded whole-body receiver.

This is development evidence, not a promotion exam. Courses here are consumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.receiving_whole_body_residual import ReceivingWholeBodyResidual
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
    asset_root: Path, policy_path: Path, course: ReceivingCourse, coefficients: tuple[float, ...]
) -> dict[str, Any]:
    mailbox = ReceiveContactMailbox(course.agent_id)
    actor = ReceivingWholeBodyResidual(
        course.agent_id, SCHEDULE.contract_hash, mailbox, coefficients
    )
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy_path,
        course=course,
        scenario_id=f"s199.rsi.r1.whole-body-search.{course.seed}",
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
    explanation = explain_receiving_window(
        trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
    )
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
        "explanation": explanation,
    }


def search(asset_root: Path, policy_path: Path, output: Path, *, candidates: int) -> dict[str, Any]:
    if output.exists() or not 2 <= candidates <= 64:
        raise ValueError("new external evidence directory and bounded candidate count required")
    root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(root):
        raise ValueError("external evidence directory required")
    source_paths = (
        "scripts/rsi_r1_whole_body_residual_v119.py",
        "src/rosclaw_soccer/rsi/receiving_whole_body_residual.py",
        "src/rosclaw_soccer/rsi/team_receive_contact_evidence.py",
        "src/rosclaw_soccer/training/receiving_experiment.py",
        "src/rosclaw_soccer/training/receiving_rollout.py",
        "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    source_hashes = {name: hash_bytes((root / name).read_bytes()) for name in source_paths}
    rng = np.random.default_rng(1190929)
    vectors: list[tuple[float, ...]] = [(0.0,) * 12]
    for i in range(1, candidates):
        # Deterministic perturbations span both contact phases and six
        # mechanically distinct synergies. No privileged trajectory fitting.
        scale = (0.25, 0.5, 0.8)[(i - 1) % 3]
        vectors.append(tuple(float(v) for v in rng.uniform(-scale, scale, 12)))
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for i, vector in enumerate(vectors):
        trials = [evaluate(asset_root, policy_path, course, vector) for course in COURSES]
        safe = all(row["safe"] and not row["physics_evidence_fault_agents"] for row in trials)
        scores = [float(row["outcome"]["shaped_return"]) for row in trials]
        row = {
            "candidate": i,
            "coefficients": vector,
            "safe": safe,
            "all_clean_first_foot": all(
                trial["first_own_foot_time_sec"] is not None and trial["prefoot_nonfoot_count"] == 0
                for trial in trials
            ),
            "controlled_count": sum(trial["outcome"]["controlled_reception"] for trial in trials),
            "mean_shaped_return": float(np.mean(scores)),
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
                    "return": row["mean_shaped_return"],
                    "failed": [t["explanation"]["failed_criteria"] for t in trials],
                }
            ),
            flush=True,
        )
    eligible = [row for row in rows if row["safe"] and row["all_clean_first_foot"]]
    best = max(eligible, key=lambda row: (row["controlled_count"], row["mean_shaped_return"]))
    parent = rows[0]
    improved = bool(
        best["controlled_count"] > parent["controlled_count"]
        or best["controlled_count"] == parent["controlled_count"]
        and best["mean_shaped_return"] > parent["mean_shaped_return"] + 0.5
    )
    report = {
        "schema": "rosclaw_soccer.rsi.r1_whole_body_residual_v119.result.v1",
        "partition": "CONSUMED_EIGHT_G1_DEVELOPMENT",
        "source_hashes": source_hashes,
        "policy_hash": hash_bytes(policy_path.read_bytes()),
        "schedule_hash": SCHEDULE.contract_hash,
        "course_count": len(COURSES),
        "candidate_count": len(rows),
        "rollout_count": len(rows) * len(COURSES),
        "parent": parent,
        "selected": best,
        "all_candidates": rows,
        "status": "DEVELOPMENT_GAIN_UNVALIDATED" if improved else "REJECTED_NO_DEVELOPMENT_GAIN",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(
        hash_bytes((root / name).read_bytes()) != digest for name, digest in source_hashes.items()
    ):
        raise ValueError("source drift during experiment")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=8)
    args = parser.parse_args()
    report = search(args.asset_root, args.policy, args.output, candidates=args.candidates)
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
