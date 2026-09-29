"""SIM_ONLY zero-authority validation of 50 Hz neural receiving in eight-G1 physics."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_course_map_v190 import COURSES
from rsi_r1_middle_basis_cem_v187 import PHYSICAL_KEYS
from rsi_r1_side_navigation_fresh_v160 import FRESH_COURSES, clean
from rsi_r1_taskspace_feedback_v125 import COURSES as ANCHOR_COURSES

from rosclaw_soccer.rsi.receiving_temporal_motor_expert import (
    ReceivingTemporalMotorExpert,
    TemporalMotorWeights,
)
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.rsi.team_receive_side_navigation import TeamReceiveSideNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_temporal_zero_v192.result.v1"
TRAIN_COURSES = (*COURSES, FRESH_COURSES[0], ANCHOR_COURSES[0])


def run_episode(
    asset_root: Path,
    policy_path: Path,
    course: ReceivingCourse,
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    slope: tuple[float, ...],
    weights: TemporalMotorWeights,
    exploration_std: float,
    exploration_seed: int,
) -> dict[str, Any]:
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingTemporalMotorExpert(
        course.agent_id,
        schedule.contract_hash,
        mailbox,
        coordination,
        left,
        right,
        left_post_gain=0.0,
        right_post_gain=0.4,
        old_lateral_slope=slope,
        far_lateral_slope=(0.0,) * 12,
        policy=weights,
        exploration_std=exploration_std,
        exploration_seed=exploration_seed,
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
        reference_policy_path=policy_path,
        course=course,
        scenario_id=f"s199.rsi.r1.coordinated-receiving.{course.seed}",
        configuration_profile="R1_CONTACT_TAP",
        oracle=schedule,
        feedback_provider=feedback,
        research_navigation_policy=navigation,
        research_contact_leg_stiffness_scale=0.4,
        research_contact_distance_threshold_m=0.24,
        research_contact_max_active_substeps=32,
        physics_evidence_consumers={course.agent_id: ReceiveContactEvidence(mailbox)},
    )
    info = result.to_dict()
    ids = tuple(sorted(row["agent_id"] for row in info["qualities"]))
    shaped, outcome = receiving_window(
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
    return {
        "course": vars(course),
        "observed_frames": feedback.observed_frames,
        "observed_features": feedback.observed_features,
        "sampled_logits": feedback.sampled_logits,
        "shaped_reward": [float(value) for value in shaped],
        "active_substeps": int(active.sum()),
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
        "physical_trace_hash": hash_json(
            {key: np.asarray(trace[key]).tolist() for key in PHYSICAL_KEYS}
        ),
    }


def _run_task(args: tuple[Any, ...]) -> dict[str, Any]:
    return run_episode(*args)


def evaluate(
    pool: ProcessPoolExecutor,
    asset_root: Path,
    policy_path: Path,
    courses: tuple[ReceivingCourse, ...],
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    slope: tuple[float, ...],
    weights: TemporalMotorWeights,
    exploration_std: float = 0.0,
    seeds: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    if seeds is None:
        seeds = tuple(0 for _ in courses)
    tasks = [
        (
            asset_root,
            policy_path,
            course,
            coordination,
            left,
            right,
            slope,
            weights,
            exploration_std,
            seed,
        )
        for course, seed in zip(courses, seeds, strict=True)
    ]
    return list(pool.map(_run_task, tasks))


def validate(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    map_report_path: Path,
    v191_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY temporal zero evidence required")
    bank_hash = preflight_receiving_courses(TRAIN_COURSES)
    parent, right_parent, refine, lateral, mapping, v191 = (
        json.loads(path.read_text())
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            map_report_path,
            v191_report_path,
        )
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, refine, lateral, mapping, v191)
    ) or (
        v191["status"] != "REJECTED_CONTEXTUAL_ACTOR_CRITIC_GATE"
        or v191["best_stage"] != "zero_parent"
        or v191["map_report_hash"] != mapping["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed rejected contextual training lineage required")
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_temporal_zero_v192.py",
            "src/rosclaw_soccer/rsi/receiving_temporal_motor_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
        rows = evaluate(
            pool,
            asset_root,
            policy_path,
            TRAIN_COURSES,
            coordination,
            left,
            right,
            slope,
            TemporalMotorWeights(),
        )
    if any(
        row["physical_trace_hash"] != old["summary"]["physical_trace_hash"]
        for row, old in zip(rows[: len(COURSES)], mapping["rows"][::4], strict=True)
    ) or not all(clean(row) and row["controlled_reception"] for row in rows[len(COURSES) :]):
        raise ValueError("temporal zero failed exact 16-course physics or retention")
    report = {
        "schema": SCHEMA,
        "v191_report_hash": v191["report_hash"],
        "map_report_hash": mapping["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_EIGHT_G1_TEMPORAL_ZERO_CONTROL",
        "course_bank_hash": bank_hash,
        "rows": rows,
        "zero_physics_equal": True,
        "retention_pass": True,
        "status": "TEMPORAL_ZERO_QUALIFIED_FOR_TRAINING",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during temporal zero validation")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--refine-report", type=Path, required=True)
    parser.add_argument("--lateral-report", type=Path, required=True)
    parser.add_argument("--map-report", type=Path, required=True)
    parser.add_argument("--v191-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.map_report,
        args.v191_report,
        args.output,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
