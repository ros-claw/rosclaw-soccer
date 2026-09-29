"""Test causal gap-rate anticipation before the measured shin-ball collision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_shin_clearance_feedback_v134 import COURSES

from rosclaw_soccer.rsi.receiving_shin_clearance_feedback import ReceivingShinClearanceFeedback
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEDULE = ReceivingOracleSchedule("red.finisher", "A1_body29", 15, 10, ((0.0,) * 29,))
# pre, post, gap gain, gap goal, foot preservation, shift cap, horizon.
PARAMETERS = (
    (0.35, 0.0, 1.0, 0.08, 0.5, 0.015, 0.0),
    (0.35, 0.0, 1.0, 0.08, 0.5, 0.015, 0.04),
    (0.35, 0.0, 1.0, 0.08, 0.5, 0.015, 0.08),
    (0.35, 0.0, 1.0, 0.08, 0.5, 0.015, 0.12),
    (0.35, 0.0, 1.0, 0.06, 0.5, 0.015, 0.08),
    (0.35, 0.0, 1.0, 0.12, 0.5, 0.015, 0.08),
    (0.35, 0.0, 1.0, 0.08, 0.8, 0.015, 0.08),
    (0.35, 0.0, 0.5, 0.08, 0.5, 0.015, 0.08),
    (0.25, 0.15, 1.0, 0.08, 0.5, 0.015, 0.08),
)


def evaluate(
    asset_root: Path,
    policy: Path,
    course: ReceivingCourse,
    parameters: tuple[float, ...],
) -> dict[str, Any]:
    pre, post, gain, clearance, preservation, shift_cap, horizon = parameters
    mailbox = ReceiveContactMailbox(course.agent_id)
    actor = ReceivingShinClearanceFeedback(
        course.agent_id,
        SCHEDULE.contract_hash,
        mailbox,
        pre,
        post,
        0.0,
        target_depth_m=0.25,
        target_lateral_m=0.12,
        clearance_gain=gain,
        target_clearance_m=clearance,
        foot_preservation=preservation,
        maximum_foot_shift_m=shift_cap,
        prediction_horizon_sec=horizon,
    )
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy,
        course=course,
        scenario_id=f"s199.rsi.r1.predictive-clearance.{course.seed}",
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
    detail = explain_receiving_window(
        trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
    )
    code = ids.index(course.agent_id) + 1
    foot = np.asarray(trace["ball_contact_agent_code"])
    effector = np.asarray(trace["ball_contact_effector_code"])
    force = np.asarray(trace["ball_contact_force_n"])
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    nf_force = np.asarray(trace["ball_nonfoot_contact_force_n"])
    first = next(
        (i for i in range(20, 120) if foot[i] == code and effector[i] in (1, 2) and force[i] > 0),
        None,
    )
    shin = (
        []
        if first is None
        else [i for i in range(first, 120) if nonfoot[i] == code and nf_force[i] > 0]
    )
    applied = np.asarray(trace["receiving_oracle_delta_rad"])
    return {
        "course": vars(course),
        "safe": info["safe"],
        "physics_evidence_fault_agents": info["physics_evidence_fault_agents"],
        "first_own_foot_time_sec": mailbox.snapshot.first_own_foot_time_sec,
        "prefoot_nonfoot_count": mailbox.snapshot.prefoot_nonfoot_count,
        "clearance_action_frames": actor.clearance_action_frames,
        "peak_predicted_foot_shift_m": actor.peak_predicted_foot_shift_m,
        "peak_actual_residual_rad": float(np.max(np.abs(applied))),
        "own_shin_frames": shin,
        "outcome": outcome,
        "explanation": detail,
        "result_hash": hash_json(info),
    }


def train(asset_root: Path, policy: Path, parent_report: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY anticipation evidence required")
    parent = json.loads(parent_report.read_text())
    if (
        parent["schema"] != "rosclaw_soccer.rsi.r1_shin_clearance_feedback_v134.result.v1"
        or parent["report_hash"]
        != hash_json({k: v for k, v in parent.items() if k != "report_hash"})
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("integrity-checked active shin feedback parent required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_predictive_clearance_v135.py",
            "src/rosclaw_soccer/rsi/receiving_shin_clearance_feedback.py",
            "src/rosclaw_soccer/skills/team/shin_clearance.py",
            "src/rosclaw_soccer/training/receiving_feedback.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rows = []
    for index, parameters in enumerate(PARAMETERS):
        trials = [evaluate(asset_root, policy, course, parameters) for course in COURSES]
        row = {
            "candidate": index,
            "parameters": parameters,
            "safe": all(t["safe"] and not t["physics_evidence_fault_agents"] for t in trials),
            "all_clean_first_foot": all(
                t["first_own_foot_time_sec"] is not None and t["prefoot_nonfoot_count"] == 0
                for t in trials
            ),
            "controlled_count": sum(t["outcome"]["controlled_reception"] for t in trials),
            "own_shin_frame_count": sum(len(t["own_shin_frames"]) for t in trials),
            "tail_distance_sum_m": sum(
                t["explanation"]["tail_maximum_foot_distance_m"] for t in trials
            ),
            "tail_speed_sum_mps": sum(
                t["explanation"]["tail_maximum_ball_speed_mps"] for t in trials
            ),
            "trials": trials,
        }
        rows.append(row)
        (output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "trials"}), flush=True)
    eligible = [row for row in rows if row["safe"] and row["all_clean_first_foot"]]
    if not eligible:
        raise ValueError("no safe first-touch-retaining anticipation; progress evidence retained")
    selected = max(
        eligible,
        key=lambda row: (
            row["controlled_count"],
            -row["own_shin_frame_count"],
            -row["tail_distance_sum_m"],
            -row["tail_speed_sum_mps"],
        ),
    )
    report = {
        "schema": "rosclaw_soccer.rsi.r1_predictive_clearance_v135.result.v1",
        "partition": "CONSUMED_EIGHT_G1_CAUSAL_ANTICIPATION_DEVELOPMENT",
        "source_hashes": sources,
        "parent_report_hash": parent["report_hash"],
        "policy_hash": hash_bytes(policy.read_bytes()),
        "candidate_count": len(rows),
        "rollout_count": len(rows) * len(COURSES),
        "parent": rows[0],
        "selected": selected,
        "all_candidates": rows,
        "status": "DEVELOPMENT_CONTROLLED_GAIN_UNVALIDATED"
        if selected["controlled_count"] > rows[0]["controlled_count"]
        else "REJECTED_NO_CONTROLLED_GAIN",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during anticipation development")
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
