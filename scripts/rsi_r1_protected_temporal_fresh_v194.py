"""Frozen SIM_ONLY hard-retention temporal actor with a genuinely new exam."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_course_map_v190 import COURSES as DEVELOPMENT_COURSES
from rsi_r1_middle_basis_cem_v187 import PHYSICAL_KEYS
from rsi_r1_temporal_actor_critic_v193 import run_episode as run_parent
from rsi_r1_temporal_zero_v192 import TRAIN_COURSES

from rosclaw_soccer.rsi.receiving_protected_temporal_expert import (
    ReceivingProtectedTemporalExpert,
)
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import TemporalMotorWeights
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

SCHEMA = "rosclaw_soccer.rsi.r1_protected_temporal_fresh_v194.result.v1"
FRESH_COURSES = (
    ReceivingCourse("red.finisher", 194001, 1.18, 0.064),
    ReceivingCourse("red.finisher", 194002, 1.30, 0.064),
    ReceivingCourse("red.finisher", 194003, 1.18, 0.068),
    ReceivingCourse("red.finisher", 194004, 1.30, 0.068),
    ReceivingCourse("red.finisher", 194005, 1.18, 0.071),
    ReceivingCourse("red.finisher", 194006, 1.30, 0.071),
    ReceivingCourse("red.finisher", 194007, 1.18, 0.076),
    ReceivingCourse("red.finisher", 194008, 1.30, 0.076),
    ReceivingCourse("red.finisher", 194009, 1.25, 0.059),
    ReceivingCourse("red.finisher", 194010, 1.25, 0.061),
    ReceivingCourse("red.finisher", 194011, 1.25, 0.079),
    ReceivingCourse("red.finisher", 194012, 1.25, 0.081),
)


def load_weights(path: Path, expected_hash: str) -> TemporalMotorWeights:
    if hash_bytes(path.read_bytes()) != expected_hash:
        raise ValueError("sealed temporal checkpoint hash required")
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {
            "input_matrix",
            "input_bias",
            "output_matrix",
            "output_bias",
        }:
            raise ValueError("four safe temporal arrays required")
        return TemporalMotorWeights(
            tuple(float(value) for value in arrays["input_matrix"].reshape(-1)),
            tuple(float(value) for value in arrays["input_bias"].reshape(-1)),
            tuple(float(value) for value in arrays["output_matrix"].reshape(-1)),
            tuple(float(value) for value in arrays["output_bias"].reshape(-1)),
        )


def run_candidate(
    asset_root: Path,
    policy_path: Path,
    course: ReceivingCourse,
    coordination: tuple[float, ...],
    left: tuple[float, ...],
    right: tuple[float, ...],
    slope: tuple[float, ...],
    weights: TemporalMotorWeights,
    protected_features: tuple[tuple[float, ...], ...],
) -> dict[str, Any]:
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingProtectedTemporalExpert(
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
        protected_initial_features=protected_features,
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
    n = len(trace["time"])
    ids = tuple(sorted(row["agent_id"] for row in info["qualities"]))
    code = ids.index(course.agent_id) + 1
    foot = np.asarray(trace["ball_contact_agent_code"])
    foot_force = np.asarray(trace["ball_contact_force_n"])
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    nonfoot_force = np.asarray(trace["ball_nonfoot_contact_force_n"])
    active = np.asarray(trace["receiving_impedance_active_substeps"], dtype=np.int64)
    if n >= 120:
        _, outcome = receiving_window(
            trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
        )
        detail = explain_receiving_window(
            trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
        )
        distance = detail["tail_maximum_foot_distance_m"]
        speed = detail["tail_maximum_ball_speed_mps"]
    else:
        outcome = {"controlled_reception": False}
        distance = 2.0
        speed = 3.0
    return {
        "course": vars(course),
        "protected_episode": feedback.protected_episode,
        "frames_recorded": n,
        "early_termination": n < 120,
        "active_substeps": int(active.sum()),
        "safe": info["safe"] and n >= 120,
        "fault_agents": info["physics_evidence_fault_agents"],
        "first_foot_frame": next(
            (
                frame
                for frame in range(20, min(n, 120))
                if foot[frame] == code and foot_force[frame] > 0
            ),
            None,
        ),
        "own_nonfoot_frames": [
            frame
            for frame in range(20, min(n, 120))
            if nonfoot[frame] == code and nonfoot_force[frame] > 0
        ],
        "controlled_reception": outcome["controlled_reception"],
        "tail_maximum_foot_distance_m": distance,
        "tail_maximum_ball_speed_mps": speed,
        "physical_trace_hash": hash_json(
            {key: np.asarray(trace[key]).tolist() for key in PHYSICAL_KEYS}
        ),
    }


def _clean(row: dict[str, Any]) -> bool:
    return bool(
        row["safe"]
        and not row["fault_agents"]
        and row["first_foot_frame"] is not None
        and not row["own_nonfoot_frames"]
        and row["active_substeps"] <= 32
    )


def _pair(args: tuple[Any, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    asset_root, policy_path, course, coordination, left, right, slope, weights, features = args
    candidate = run_candidate(
        asset_root, policy_path, course, coordination, left, right, slope, weights, features
    )
    old = run_parent(
        asset_root,
        policy_path,
        course,
        coordination,
        left,
        right,
        slope,
        TemporalMotorWeights(),
        0.0,
        0,
    )
    return candidate, old


def exam(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    zero_report_path: Path,
    v193_report_path: Path,
    checkpoint: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY protected fresh evidence required")
    fresh_bank_hash = preflight_receiving_courses(FRESH_COURSES)
    parent, right_parent, refine, lateral, zero, v193 = (
        json.loads(path.read_text())
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            zero_report_path,
            v193_report_path,
        )
    )
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, right_parent, refine, lateral, zero, v193)
    ) or (
        v193["status"] != "REJECTED_TEMPORAL_ACTOR_CRITIC_GATE"
        or v193["zero_report_hash"] != zero["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
        or v193["history"][1]["deterministic_score"][:2] != [0, 3]
    ):
        raise ValueError("sealed temporal development and zero lineage required")
    weights = load_weights(checkpoint, v193["history"][1]["checkpoint_hash"])
    protected_features = tuple(tuple(row["observed_features"][0]) for row in zero["rows"][-2:])
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(
        float(v) for v in np.asarray(right_parent["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_protected_temporal_fresh_v194.py",
            "src/rosclaw_soccer/rsi/receiving_protected_temporal_expert.py",
            "src/rosclaw_soccer/rsi/receiving_temporal_motor_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
        development = list(
            pool.map(
                run_candidate,
                (asset_root,) * len(TRAIN_COURSES),
                (policy_path,) * len(TRAIN_COURSES),
                TRAIN_COURSES,
                (coordination,) * len(TRAIN_COURSES),
                (left,) * len(TRAIN_COURSES),
                (right,) * len(TRAIN_COURSES),
                (slope,) * len(TRAIN_COURSES),
                (weights,) * len(TRAIN_COURSES),
                (protected_features,) * len(TRAIN_COURSES),
            )
        )
        retention = development[len(DEVELOPMENT_COURSES) :]
        retention_equal = all(
            row["protected_episode"] and row["physical_trace_hash"] == old["physical_trace_hash"]
            for row, old in zip(retention, zero["rows"][-2:], strict=True)
        )
        development_pass = sum(
            _clean(row) and row["controlled_reception"]
            for row in development[: len(DEVELOPMENT_COURSES)]
        )
        candidate_rows: list[dict[str, Any]] = []
        parent_rows: list[dict[str, Any]] = []
        if retention_equal and development_pass >= 3:
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
                    protected_features,
                )
                for course in FRESH_COURSES
            ]
            for index, (candidate, old) in enumerate(pool.map(_pair, tasks)):
                candidate_rows.append(candidate)
                parent_rows.append(old)
                (output / "progress.json").write_text(
                    json.dumps({"candidate": candidate_rows, "parent": parent_rows}, indent=2)
                    + "\n"
                )
                print(
                    json.dumps(
                        {
                            "case": index,
                            "candidate": _clean(candidate) and candidate["controlled_reception"],
                            "parent": _clean(old) and old["controlled_reception"],
                            "protected": candidate["protected_episode"],
                            "early": candidate["early_termination"],
                        }
                    ),
                    flush=True,
                )
    candidate_pass = sum(_clean(row) and row["controlled_reception"] for row in candidate_rows)
    parent_pass = sum(_clean(row) and row["controlled_reception"] for row in parent_rows)
    qualified = (
        retention_equal
        and development_pass >= 3
        and candidate_pass == len(FRESH_COURSES)
        and candidate_pass > parent_pass
    )
    report = {
        "schema": SCHEMA,
        "v193_report_hash": v193["report_hash"],
        "source_hashes": sources,
        "partition": "FROZEN_NEW_PROTECTED_TEMPORAL_RECEIVING_HOLDOUT",
        "fresh_bank_hash": fresh_bank_hash,
        "fresh_courses": [vars(course) for course in FRESH_COURSES],
        "checkpoint_hash": v193["history"][1]["checkpoint_hash"],
        "protection_radius": 0.006,
        "protected_feature_hash": hash_json(protected_features),
        "development": development,
        "development_pass": development_pass,
        "retention_equal": retention_equal,
        "candidate": candidate_rows,
        "parent": parent_rows,
        "candidate_pass": candidate_pass,
        "parent_pass": parent_pass,
        "status": "QUALIFIED_LOCAL_PROTECTED_TEMPORAL_ONLY"
        if qualified
        else "REJECTED_PROTECTED_TEMPORAL_FRESH_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during protected temporal exam")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--refine-report", type=Path, required=True)
    parser.add_argument("--lateral-report", type=Path, required=True)
    parser.add_argument("--zero-report", type=Path, required=True)
    parser.add_argument("--v193-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = exam(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.zero_report,
        args.v193_report,
        args.checkpoint,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "development_pass": report["development_pass"],
                "candidate_pass": report["candidate_pass"],
                "parent_pass": report["parent_pass"],
                "report_hash": report["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
