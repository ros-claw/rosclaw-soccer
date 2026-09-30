"""SIM_ONLY velocity-safe near-side motor retention and untouched new exam."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_contact_manifold_v222 import context
from rsi_r1_foot_servo_contact_v235 import _run as _run_parent
from rsi_r1_gated_nearside_fresh_v237 import TRAIN_SEEDS
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_middle_basis_cem_v187 import PHYSICAL_KEYS
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy

from rosclaw_soccer.rsi.receiving_velocity_gated_precontact_motor import (
    ReceivingVelocityGatedPrecontactMotor,
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

SCHEMA = "rosclaw_soccer.rsi.r1_velocity_gated_fresh_v238.result.v1"
FRESH_COURSES = tuple(
    ReceivingCourse("red.finisher", 238001 + i * 4 + j, speed, lateral)
    for i, lateral in enumerate((0.064, 0.065))
    for j, speed in enumerate((1.18, 1.20, 1.22, 1.24))
)
THRESHOLD = -0.48
RADIUS = 0.017
ANCHOR = 158


def _run_velocity(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        asset_root,
        policy_path,
        course,
        coordination,
        left,
        right,
        slope,
        knots,
        references,
        low_weights,
        protected,
        pre_weights,
        post_weights,
        candidate_weights,
        states,
    ) = task
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingVelocityGatedPrecontactMotor(
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
        skill_knots=knots,
        references=references,
        corrected_experts=("high",),
        foot_gain=0.3,
        post_multiplier=1.0,
        low_post_weights=low_weights,
        neural_weights=pre_weights,
        protected_initial_features=protected,
        post_policy=post_weights,
        candidate_neural_weights=candidate_weights,
        activation_states=states,
        activation_radius=RADIUS,
        minimum_relative_vx_feature=THRESHOLD,
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
    n = len(trace["time"])
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
    code = ids.index(course.agent_id) + 1
    foot = np.asarray(trace["ball_contact_agent_code"])
    foot_force = np.asarray(trace["ball_contact_force_n"])
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    nonfoot_force = np.asarray(trace["ball_nonfoot_contact_force_n"])
    active = np.asarray(trace["receiving_impedance_active_substeps"], dtype=np.int64)
    return {
        "course": vars(course),
        "selected_expert": feedback.selected_expert,
        "protected_episode": feedback.protected_episode,
        "activation_decided": feedback.activation_decided,
        "activation_enabled": feedback.activation_enabled,
        "activation_features": feedback.activation_features,
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
        "active_substeps": int(active.sum()),
        "controlled_reception": outcome["controlled_reception"],
        "tail_maximum_foot_distance_m": distance,
        "tail_maximum_ball_speed_mps": speed,
        "physical_trace_hash": hash_json(
            {key: np.asarray(trace[key]).tolist() for key in PHYSICAL_KEYS}
        ),
    }


def exam(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY velocity-gated evidence required")
    fresh_hash = preflight_receiving_courses(FRESH_COURSES)
    v223, v224, v229, v232, v233, v235, v236, v237 = (
        _checked(path)
        for path in (
            args.v223_report,
            args.v224_report,
            args.v229_dir / "report.json",
            args.v232_dir / "report.json",
            args.v233_dir / "report.json",
            args.v235_dir / "report.json",
            args.v236_dir / "report.json",
            args.v237_dir / "report.json",
        )
    )
    if (
        v237["status"] != "REJECTED_GATED_NEARSIDE_FRESH_GATE"
        or v237["v236_report_hash"] != v236["report_hash"]
        or v237["old_candidate_score"][:2] != [7, 9]
        or v237["fresh_candidate_score"][:2] != [4, 6]
        or v236["v235_report_hash"] != v235["report_hash"]
        or v233["v232_report_hash"] != v232["report_hash"]
        or v224["v223_report_hash"] != v223["report_hash"]
    ):
        raise ValueError("sealed fast-ball safety failure required")
    common, (base_weights, _, _), lineage = context(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.v205_dir,
        args.map_report,
        args.v214_report,
        args.v215_dir,
        args.v216_report,
        args.v218_dir,
        args.v220_report,
    )
    if lineage["v220_report_hash"] != v224["v220_report_hash"]:
        raise ValueError("sealed precontact lineage required")
    base_bias = np.asarray(base_weights.output_bias) + np.asarray(v224["best_offset"])
    pre_weights = replace(base_weights, output_bias=tuple(float(x) for x in base_bias))
    candidate_weights = replace(
        base_weights,
        output_bias=tuple(float(x) for x in base_bias + np.asarray(v236["best_offset"])),
    )
    neural = _load_policy(args.v229_dir / "update-1.npz", v229["history"][0]["checkpoint_hash"])
    anchor = next(row for row in v233["candidates"] if row["source_episode"] == ANCHOR)
    post_weights = replace(
        neural,
        output_bias=tuple(
            float(x) for x in np.asarray(neural.output_bias) + np.asarray(anchor["latent_offset"])
        ),
    )
    courses = tuple(ReceivingCourse(**row["course"]) for row in v232["course_summary"])
    states = tuple(tuple(float(x) for x in row) for row in v237["activation_states"])
    if len(courses) != 11 or len(states) != 3:
        raise ValueError("sealed eleven-course measured activation states required")

    def parent_tasks(bank: tuple[ReceivingCourse, ...]) -> list[tuple[Any, ...]]:
        return [
            (common[0], common[1], course, *common[2:], pre_weights, post_weights, 0.0, 0.0)
            for course in bank
        ]

    def candidate_tasks(bank: tuple[ReceivingCourse, ...]) -> list[tuple[Any, ...]]:
        return [
            (
                common[0],
                common[1],
                course,
                *common[2:],
                pre_weights,
                post_weights,
                candidate_weights,
                states,
            )
            for course in bank
        ]

    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_velocity_gated_fresh_v238.py",
            "src/rosclaw_soccer/rsi/receiving_velocity_gated_precontact_motor.py",
            "src/rosclaw_soccer/rsi/receiving_gated_precontact_motor.py",
            "scripts/rsi_r1_foot_servo_contact_v235.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        old_parent = list(pool.map(_run_parent, parent_tasks(courses)))
        if any(
            row["physical_trace_hash"] != sealed["physical_trace_hash"]
            for row, sealed in zip(old_parent, v235["baseline"], strict=True)
        ):
            raise ValueError("parent must replay sealed broad physical evidence")
        old_candidate = list(pool.map(_run_velocity, candidate_tasks(courses)))
        old_unchanged = all(
            row["physical_trace_hash"] == old["physical_trace_hash"]
            for row, old in zip(old_candidate, old_parent, strict=True)
            if row["course"]["seed"] not in TRAIN_SEEDS
        )
        retained = all(
            not (clean(old) and old["controlled_reception"])
            or (clean(row) and row["controlled_reception"])
            for old, row in zip(old_parent, old_candidate, strict=True)
        )
        eligible = (
            old_unchanged
            and retained
            and _score(old_candidate)[0] >= _score(old_parent)[0] + 2
            and _score(old_candidate)[1] >= _score(old_parent)[1]
        )
        if eligible:
            consumed_parent = list(
                pool.map(
                    _run_parent,
                    parent_tasks(tuple(ReceivingCourse(**row) for row in v237["fresh_courses"])),
                )
            )
            consumed_candidate = list(
                pool.map(
                    _run_velocity,
                    candidate_tasks(tuple(ReceivingCourse(**row) for row in v237["fresh_courses"])),
                )
            )
            safety_repaired = (
                _score(consumed_candidate)[0] >= _score(consumed_parent)[0] + 2
                and _score(consumed_candidate)[1] >= _score(consumed_parent)[1]
            )
            if safety_repaired:
                fresh_parent = list(pool.map(_run_parent, parent_tasks(FRESH_COURSES)))
                fresh_candidate = list(pool.map(_run_velocity, candidate_tasks(FRESH_COURSES)))
            else:
                fresh_parent = []
                fresh_candidate = []
        else:
            consumed_parent = []
            consumed_candidate = []
            fresh_parent = []
            fresh_candidate = []
            safety_repaired = False
    consumed_parent_score = _score(consumed_parent) if eligible else None
    consumed_candidate_score = _score(consumed_candidate) if eligible else None
    fresh_parent_score = _score(fresh_parent) if eligible else None
    fresh_candidate_score = _score(fresh_candidate) if eligible else None
    qualified = bool(
        eligible
        and safety_repaired
        and fresh_parent_score is not None
        and fresh_candidate_score is not None
        and fresh_candidate_score[0] >= 4
        and fresh_candidate_score[1] >= 7
        and fresh_candidate_score[0] >= fresh_parent_score[0] + 2
    )
    report = {
        "schema": SCHEMA,
        "v237_report_hash": v237["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_SPEED_SAFETY_DIAGNOSIS_THEN_NEW_FRESH8",
        "minimum_relative_vx_feature": THRESHOLD,
        "activation_radius": RADIUS,
        "fresh_bank_hash": fresh_hash,
        "fresh_courses": [vars(course) for course in FRESH_COURSES],
        "old_parent_score": _score(old_parent),
        "old_candidate_score": _score(old_candidate),
        "old_candidate": old_candidate,
        "old_unchanged_elsewhere": old_unchanged,
        "old_success_retained": retained,
        "consumed_parent_score": consumed_parent_score,
        "consumed_candidate_score": consumed_candidate_score,
        "consumed_candidate": consumed_candidate,
        "consumed_safety_repaired": safety_repaired,
        "fresh_parent_score": fresh_parent_score,
        "fresh_candidate_score": fresh_candidate_score,
        "fresh_parent": fresh_parent,
        "fresh_candidate": fresh_candidate,
        "status": "FRESH_VELOCITY_SAFE_NEARSIDE_QUALIFIED_FOR_NEXT_CHAIN_GATE"
        if qualified
        else "REJECTED_VELOCITY_SAFE_NEARSIDE_FRESH_GATE"
        if safety_repaired
        else "REJECTED_VELOCITY_SAFE_NEARSIDE_DEVELOPMENT_GATE"
        if eligible
        else "REJECTED_VELOCITY_SAFE_NEARSIDE_RETENTION_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during velocity-safe exam")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "asset-root",
        "policy",
        "parent-report",
        "right-report",
        "refine-report",
        "lateral-report",
        "v205-dir",
        "map-report",
        "v214-report",
        "v215-dir",
        "v216-report",
        "v218-dir",
        "v220-report",
        "v223-report",
        "v224-report",
        "v229-dir",
        "v232-dir",
        "v233-dir",
        "v235-dir",
        "v236-dir",
        "v237-dir",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    report = exam(parser.parse_args())
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "old_parent_score",
                    "old_candidate_score",
                    "consumed_parent_score",
                    "consumed_candidate_score",
                    "fresh_parent_score",
                    "fresh_candidate_score",
                    "report_hash",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
