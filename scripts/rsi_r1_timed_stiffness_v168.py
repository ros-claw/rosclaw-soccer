"""SIM_ONLY finite-substep focal contact stiffness around measured approach."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_distance_stiffness_v167 import PHYSICAL_KEYS
from rsi_r1_distance_stiffness_v167 import run as run_parent
from rsi_r1_left_positive_mining_v163 import rank
from rsi_r1_side_navigation_fresh_v160 import FRESH_COURSES, clean

from rosclaw_soccer.rsi.receiving_side_conditioned_expert import ReceivingSideConditionedExpert
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.rsi.team_receive_side_navigation import TeamReceiveSideNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_timed_stiffness_v168.result.v1"
SETTINGS = tuple(
    (int(cap), float(scale)) for cap in (20, 40, 60, 100) for scale in (0.85, 0.7, 0.55, 0.4)
)


def run(
    asset_root: Path,
    policy: Path,
    course: ReceivingCourse,
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    stiffness_scale: float,
    max_active_substeps: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingSideConditionedExpert(
        course.agent_id,
        schedule.contract_hash,
        mailbox,
        coordination,
        left,
        right,
        left_post_gain=0.0,
        right_post_gain=0.4,
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
        research_contact_leg_stiffness_scale=stiffness_scale,
        research_contact_distance_threshold_m=0.24,
        research_contact_max_active_substeps=max_active_substeps,
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
    summary = {
        "course": vars(course),
        "stiffness_scale": stiffness_scale,
        "max_active_substeps": max_active_substeps,
        "active_substeps": int(active.sum()),
        "active_frames": np.flatnonzero(active > 0).tolist(),
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
    return summary, trace


def train(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    right_report: Path,
    mining_report: Path,
    distance_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY timed stiffness evidence required")
    parent, right_parent, mining, distance = (
        json.loads(path.read_text())
        for path in (parent_report, right_report, mining_report, distance_report)
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, mining, distance)
    ) or (
        distance["status"] != "REJECTED_FOCAL_IMPEDANCE_CONTROLLED_GATE"
        or not distance["zero_physics_equal"]
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed focal distance-gate lineage required")
    course = FRESH_COURSES[mining["course_index"]]
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(mining["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_timed_stiffness_v168.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    baseline, baseline_trace = run_parent(
        asset_root, policy, course, coordination, left, right, 1.0, None
    )
    zero, zero_trace = run(asset_root, policy, course, coordination, left, right, 0.7, 0)
    zero_physics_equal = all(
        key in baseline_trace
        and key in zero_trace
        and np.array_equal(baseline_trace[key], zero_trace[key])
        for key in PHYSICAL_KEYS
    )
    rows = []
    if zero_physics_equal:
        for cap, scale in SETTINGS:
            row, _ = run(asset_root, policy, course, coordination, left, right, scale, cap)
            if row["active_substeps"] > cap:
                raise ValueError("SIM_ONLY stiffness duration guard exceeded")
            rows.append(row)
            (output / "progress.json").write_text(
                json.dumps(rows, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {
                        "cap": cap,
                        "scale": scale,
                        "active_substeps": row["active_substeps"],
                        "active_frames": row["active_frames"],
                        "clean": clean(row),
                        "controlled": row["controlled_reception"],
                        "nonfoot": row["own_nonfoot_frames"],
                        "distance": row["tail_maximum_foot_distance_m"],
                        "speed": row["tail_maximum_ball_speed_mps"],
                    }
                ),
                flush=True,
            )
    active_rows = [row for row in rows if row["active_substeps"] > 0]
    best = max(active_rows, key=rank) if active_rows else None
    check = (
        run(
            asset_root,
            policy,
            course,
            coordination,
            left,
            right,
            best["stiffness_scale"],
            best["max_active_substeps"],
        )[0]
        if best is not None
        else None
    )
    positive = check is not None and clean(check) and check["controlled_reception"]
    report = {
        "schema": SCHEMA,
        "distance_report_hash": distance["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_TIMED_FOCAL_STIFFNESS",
        "course": vars(course),
        "zero_physics_equal": zero_physics_equal,
        "zero_physics_keys": PHYSICAL_KEYS,
        "baseline": baseline,
        "zero": zero,
        "rows": rows,
        "best": best,
        "check": check,
        "status": "DEVELOPMENT_TIMED_IMPEDANCE_FEASIBLE_UNVALIDATED"
        if positive and zero_physics_equal
        else "REJECTED_TIMED_IMPEDANCE_CONTROLLED_GATE"
        if zero_physics_equal and active_rows
        else "REJECTED_ZERO_OR_NO_ACTIVATION_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during timed stiffness feasibility")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--mining-report", type=Path, required=True)
    parser.add_argument("--distance-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.mining_report,
        args.distance_report,
        args.output,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
