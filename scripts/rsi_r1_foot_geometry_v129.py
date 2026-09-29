"""Measured eight-G1 development search over post-touch foot placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.receiving_taskspace_motor import ReceivingTaskspaceMotor
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

COURSES = (
    ReceivingCourse("red.finisher", 92801, 1.25, 0.08),
    ReceivingCourse("red.finisher", 92803, 1.5, -0.08),
)
PARAMETERS = (
    (0.0, 0.18, 0.03),
    (0.25, 0.18, 0.03),
    (0.25, 0.18, 0.10),
    (0.25, 0.12, 0.10),
    (0.25, 0.25, 0.10),
    (0.5, 0.18, 0.10),
    (0.5, 0.12, 0.10),
    (0.5, 0.18, -0.03),
    (0.5, 0.25, -0.03),
)


def evaluate(
    asset_root: Path,
    policy: Path,
    course: ReceivingCourse,
    gain: float,
    depth: float,
    lateral: float,
) -> dict[str, Any]:
    mailbox = ReceiveContactMailbox(course.agent_id)
    motor = ReceivingTaskspaceMotor(
        course.agent_id,
        mailbox,
        0.0,
        gain,
        0.0,
        idle_before_first_touch=True,
        target_depth_m=depth,
        target_lateral_m=lateral,
    )
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy,
        course=course,
        scenario_id=f"s199.rsi.r1.foot-geometry.{course.seed}",
        configuration_profile="R1_CONTACT_TAP",
        research_motor_option=motor,
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
    contact = np.asarray(trace["ball_contact_agent_code"])
    effector = np.asarray(trace["ball_contact_effector_code"])
    force = np.asarray(trace["ball_contact_force_n"])
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    nonfoot_force = np.asarray(trace["ball_nonfoot_contact_force_n"])
    code = ids.index(course.agent_id) + 1
    first = next(
        (
            i
            for i in range(20, 120)
            if contact[i] == code and effector[i] in (1, 2) and force[i] > 0
        ),
        None,
    )
    shin = (
        []
        if first is None
        else [i for i in range(first, 120) if nonfoot[i] == code and nonfoot_force[i] > 0]
    )
    return {
        "course": vars(course),
        "safe": info["safe"],
        "motor_fault_agents": info["motor_fault_agents"],
        "physics_evidence_fault_agents": info["physics_evidence_fault_agents"],
        "first_own_foot_time_sec": mailbox.snapshot.first_own_foot_time_sec,
        "prefoot_nonfoot_count": mailbox.snapshot.prefoot_nonfoot_count,
        "motor_nonzero_target_frames": motor.nonzero_target_frames,
        "motor_peak_correction_rad": motor.peak_correction_rad,
        "own_shin_frames": shin,
        "outcome": outcome,
        "explanation": explanation,
        "result_hash": hash_json(info),
    }


def train(asset_root: Path, policy: Path, parent_report: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY evidence required")
    parent = json.loads(parent_report.read_text())
    if (
        parent["schema"] != "rosclaw_soccer.rsi.r1_posttouch_motor_v128.result.v1"
        or parent["report_hash"]
        != hash_json({k: v for k, v in parent.items() if k != "report_hash"})
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("integrity-checked posttouch parent required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_foot_geometry_v129.py",
            "src/rosclaw_soccer/rsi/receiving_taskspace_motor.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/training/receiving_rollout.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    parent_touch = {
        t["course"]["seed"]: t["first_own_foot_time_sec"] for t in parent["parent"]["trials"]
    }
    output.mkdir(parents=True)
    rows = []
    for index, (gain, depth, lateral) in enumerate(PARAMETERS):
        trials = [evaluate(asset_root, policy, course, gain, depth, lateral) for course in COURSES]
        row = {
            "candidate": index,
            "post_gain": gain,
            "target_depth_m": depth,
            "target_lateral_m": lateral,
            "safe": all(
                t["safe"] and not t["motor_fault_agents"] and not t["physics_evidence_fault_agents"]
                for t in trials
            ),
            "first_touch_retained": all(
                t["first_own_foot_time_sec"] is not None
                and abs(t["first_own_foot_time_sec"] - parent_touch[t["course"]["seed"]])
                <= 0.002001
                and t["prefoot_nonfoot_count"] == 0
                for t in trials
            ),
            "controlled_count": sum(t["outcome"]["controlled_reception"] for t in trials),
            "own_shin_frame_count": sum(len(t["own_shin_frames"]) for t in trials),
            "tail_distance_sum_m": sum(
                t["explanation"]["tail_maximum_foot_distance_m"] for t in trials
            ),
            "trials": trials,
        }
        rows.append(row)
        (output / "progress.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "trials"}), flush=True)
    eligible = [r for r in rows if r["safe"] and r["first_touch_retained"]]
    if not eligible:
        raise ValueError("no safe first-touch-retaining candidate; progress evidence retained")
    selected = max(
        eligible,
        key=lambda r: (
            r["controlled_count"],
            -r["own_shin_frame_count"],
            -r["tail_distance_sum_m"],
        ),
    )
    report = {
        "schema": "rosclaw_soccer.rsi.r1_foot_geometry_v129.result.v1",
        "partition": "CONSUMED_EIGHT_G1_FOOT_GEOMETRY_DEVELOPMENT",
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
        raise ValueError("source drift during foot geometry development")
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
