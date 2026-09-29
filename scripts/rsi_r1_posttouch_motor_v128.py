"""Train bounded post-first-foot motor handoff with eight-G1 physical credit."""

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
    (0.0, 0.0, 0.0, 0.0),
    (0.25, 0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0, 0.0),
    (0.75, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.5, 0.05, 0.0, 0.0),
    (0.5, 0.1, 0.0, 0.0),
    (0.75, 0.05, 0.0, 0.0),
    (0.5, 0.05, 0.03, -0.05),
)


def evaluate(
    asset_root: Path,
    policy: Path,
    course: ReceivingCourse,
    parameters: tuple[float, float, float, float],
) -> dict[str, Any]:
    mailbox = ReceiveContactMailbox(course.agent_id)
    post_gain, horizon, hip, ankle = parameters
    motor = ReceivingTaskspaceMotor(
        course.agent_id,
        mailbox,
        0.0,
        post_gain,
        horizon,
        hip,
        ankle,
        idle_before_first_touch=True,
    )
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy,
        course=course,
        scenario_id=f"s199.rsi.r1.posttouch-motor.{course.seed}",
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
    contact_force = np.asarray(trace["ball_contact_force_n"])
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    nonfoot_force = np.asarray(trace["ball_nonfoot_contact_force_n"])
    code = ids.index(course.agent_id) + 1
    first = next(
        (
            i
            for i in range(20, 120)
            if contact[i] == code and effector[i] in (1, 2) and contact_force[i] > 0
        ),
        None,
    )
    shin_frames = (
        []
        if first is None
        else [i for i in range(first, 120) if nonfoot[i] == code and nonfoot_force[i] > 0]
    )
    return {
        "course": vars(course),
        "safe": info["safe"],
        "motor_fault_agents": info["motor_fault_agents"],
        "physics_evidence_fault_agents": info["physics_evidence_fault_agents"],
        "result_hash": hash_json(info),
        "control_frames": len(trace["time"]),
        "first_own_foot_time_sec": mailbox.snapshot.first_own_foot_time_sec,
        "prefoot_nonfoot_count": mailbox.snapshot.prefoot_nonfoot_count,
        "motor_nonzero_target_frames": motor.nonzero_target_frames,
        "motor_peak_correction_rad": motor.peak_correction_rad,
        "own_shin_frames": shin_frames,
        "outcome": outcome,
        "explanation": explanation,
    }


def train(asset_root: Path, policy: Path, handoff_report: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY motor evidence required")
    handoff = json.loads(handoff_report.read_text())
    if (
        handoff["schema"] != "rosclaw_soccer.rsi.r1_idle_motor_handoff_v127.result.v1"
        or handoff["status"] != "IDLE_MOTOR_NONINTERFERENCE_QUALIFIED"
        or handoff["report_hash"]
        != hash_json({key: value for key, value in handoff.items() if key != "report_hash"})
        or handoff["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("qualified idle motor ownership baseline required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_posttouch_motor_v128.py",
            "src/rosclaw_soccer/rsi/receiving_taskspace_motor.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/training/receiving_rollout.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    parent_contacts = {
        row["course"]["seed"]: row["first_own_foot_time_sec"] for row in handoff["courses"]
    }
    output.mkdir(parents=True)
    rows = []
    for i, parameters in enumerate(PARAMETERS):
        trials = [evaluate(asset_root, policy, course, parameters) for course in COURSES]
        first_touch_retained = all(
            t["first_own_foot_time_sec"] is not None
            and abs(t["first_own_foot_time_sec"] - parent_contacts[t["course"]["seed"]]) <= 0.002001
            and t["prefoot_nonfoot_count"] == 0
            for t in trials
        )
        row = {
            "candidate": i,
            "parameters": parameters,
            "safe": all(
                t["safe"] and not t["motor_fault_agents"] and not t["physics_evidence_fault_agents"]
                for t in trials
            ),
            "first_touch_retained": first_touch_retained,
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
        print(
            json.dumps(
                {
                    "candidate": i,
                    "safe": row["safe"],
                    "first_touch_retained": first_touch_retained,
                    "controlled": row["controlled_count"],
                    "shin": row["own_shin_frame_count"],
                    "tail_distance_sum_m": row["tail_distance_sum_m"],
                }
            ),
            flush=True,
        )
    eligible = [row for row in rows if row["safe"] and row["first_touch_retained"]]
    if not eligible:
        raise ValueError("no safe first-touch-retaining motor; progress evidence retained")
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
        "schema": "rosclaw_soccer.rsi.r1_posttouch_motor_v128.result.v1",
        "partition": "CONSUMED_EIGHT_G1_MOTOR_DEVELOPMENT",
        "source_hashes": sources,
        "handoff_report_hash": handoff["report_hash"],
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
        raise ValueError("source drift during posttouch motor development")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--handoff-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train(args.asset_root, args.policy, args.handoff_report, args.output)
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
