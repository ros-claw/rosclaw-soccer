"""SIM_ONLY state-routed composition and predeclared fresh receiving examination."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_course_map_v190 import _find_candidate
from rsi_r1_measured_skill_router_v207 import FRESH_COURSES as CONSUMED_COURSES
from rsi_r1_measured_skill_router_v207 import _score
from rsi_r1_middle_basis_cem_v187 import PHYSICAL_KEYS
from rsi_r1_temporal_zero_v192 import TRAIN_COURSES

from rosclaw_soccer.rsi.receiving_composed_contact_router import ReceivingComposedContactRouter
from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
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

SCHEMA = "rosclaw_soccer.rsi.r1_composed_contact_fresh_v214.result.v1"
FRESH_COURSES = tuple(
    ReceivingCourse("red.finisher", 214001 + i * 4 + j, speed, lateral)
    for i, lateral in enumerate((0.065, 0.071))
    for j, speed in enumerate((1.17, 1.24, 1.30, 1.33))
)


def _run_task(task: tuple[Any, ...]) -> dict[str, Any]:
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
    ) = task
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingComposedContactRouter(
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
        "selected_distance": feedback.selected_distance,
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
        "active_substeps": int(active.sum()),
        "tail_maximum_foot_distance_m": distance,
        "tail_maximum_ball_speed_mps": speed,
        "foot_feedback_frames": feedback.nonzero_frames,
        "low_support_frames": feedback.low_active_frames,
        "physical_trace_hash": hash_json(
            {key: np.asarray(trace[key]).tolist() for key in PHYSICAL_KEYS}
        ),
    }


def exam(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    router_report_path: Path,
    v187_report: Path,
    v188_report: Path,
    map_report_path: Path,
    v205_dir: Path,
    v208_dir: Path,
    v209_report: Path,
    v212_report: Path,
    v213_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY composed contact evidence required")
    development_hash = preflight_receiving_courses(CONSUMED_COURSES)
    fresh_hash = preflight_receiving_courses(FRESH_COURSES)
    (
        parent,
        right_parent,
        refine,
        lateral,
        router,
        v187,
        v188,
        mapping,
        v205,
        v208,
        v209,
        v212,
        v213,
    ) = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            router_report_path,
            v187_report,
            v188_report,
            map_report_path,
            v205_dir / "report.json",
            v208_dir / "report.json",
            v209_report,
            v212_report,
            v213_report,
        )
    )
    if (
        router["status"] != "REJECTED_MEASURED_SKILL_ROUTER_GATE"
        or router["fresh_bank_hash"] != development_hash
        or v208["router_report_hash"] != router["report_hash"]
        or v209["scores"][1][:2] != [1, 7]
        or v212["oracle_coverage"] != 2
        or v187["status"] != "REJECTED_MIDDLE_TRAINING_GATE"
        or v205["map_report_hash"] != mapping["report_hash"]
        or v213["status"] != "REJECTED_POST_CONTACT_CEM_GATE"
        or not v213["finalists"][0]["anchor_physical_equal"]
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed measured specialist development lineage required")
    center_features_path = v208_dir / "fresh-207006.npz"
    center_row = next(row for row in v208["rows"] if row["course"]["seed"] == 207006)
    if hash_bytes(center_features_path.read_bytes()) != center_row["feature_hash"]:
        raise ValueError("sealed center-skill measured initial state required")
    with np.load(center_features_path, allow_pickle=False) as arrays:
        center_features = np.asarray(arrays["features"], dtype=np.float64)
    knots = tuple(
        ReceivingSkillKnot(
            float(row["lateral_m"]),
            float(row["closing_speed_mps"]),
            row["expert"],
            tuple(float(value) for value in row["weights"]),
        )
        for row in router["skill_knots"]
    ) + (
        ReceivingSkillKnot(
            float(center_features[0, 1] * 0.2),
            float(center_features[0, 3] * 2.0),
            "center",
            tuple(float(value) for value in _find_candidate(v188, 2, 7)["middle_weights"]),
        ),
    )
    references = []
    for index, row in enumerate(v205["rows"]):
        if row["expert"] != "high":
            continue
        teacher_path = v205_dir / f"teacher-{index}.npz"
        if hash_bytes(teacher_path.read_bytes()) != row["feature_hash"]:
            raise ValueError("sealed high teacher required")
        with np.load(teacher_path, allow_pickle=False) as arrays:
            features = np.asarray(arrays["features"], dtype=np.float64)
        historical = next(
            item["summary"]
            for item in mapping["rows"]
            if item["expert"] == "high"
            and item["summary"]["course"]["seed"] == row["course"]["seed"]
        )
        references.append(
            ReceivingFootPhaseReference(
                "high",
                float(features[0, 1] * 0.2),
                float(features[0, 3] * 2.0),
                int(historical["first_foot_frame"]),
                tuple(tuple(float(value) for value in frame) for frame in features),
            )
        )
    if len(references) != 3:
        raise ValueError("three genuine high teacher traces required")
    low_weights = tuple(float(value) for value in v213["finalists"][0]["weights"])
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_composed_contact_fresh_v214.py",
            "src/rosclaw_soccer/rsi/receiving_composed_contact_router.py",
            "src/rosclaw_soccer/rsi/receiving_foot_phase_router.py",
            "src/rosclaw_soccer/rsi/receiving_measured_skill_router.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)

    def tasks(courses: tuple[ReceivingCourse, ...]) -> list[tuple[Any, ...]]:
        return [
            (
                asset_root,
                policy_path,
                course,
                coordination,
                left,
                right,
                slope,
                knots,
                tuple(references),
                low_weights,
            )
            for course in courses
        ]

    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        development = list(pool.map(_run_task, tasks(CONSUMED_COURSES)))
        anchors = list(pool.map(_run_task, tasks(TRAIN_COURSES[-2:])))
        anchor_equal = all(
            row["selected_expert"] == "parent"
            and row["physical_trace_hash"] == old["physical_trace_hash"]
            for row, old in zip(anchors, router["development"][-2:], strict=True)
        )
        development_score = _score(development)
        development_gate = anchor_equal and development_score[0] >= 4 and development_score[1] >= 7
        fresh = list(pool.map(_run_task, tasks(FRESH_COURSES))) if development_gate else []
    fresh_score = _score(fresh) if fresh else None
    qualified = (
        development_gate and fresh_score is not None and fresh_score[0] >= 6 and fresh_score[1] >= 7
    )
    report = {
        "schema": SCHEMA,
        "router_report_hash": router["report_hash"],
        "v213_report_hash": v213["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_DEVELOPMENT_THEN_PREDECLARED_NEW_FRESH",
        "development_bank_hash": development_hash,
        "fresh_bank_hash": fresh_hash,
        "fresh_courses": [vars(course) for course in FRESH_COURSES],
        "knots": [vars(knot) for knot in knots],
        "low_weights": low_weights,
        "development": development,
        "development_score": development_score,
        "anchors": anchors,
        "anchor_physical_equal": anchor_equal,
        "development_gate": development_gate,
        "fresh_candidate": fresh,
        "fresh_candidate_score": fresh_score,
        "status": "FRESH_COMPOSED_CONTACT_QUALIFIED_FOR_NEXT_CHAIN_GATE"
        if qualified
        else "REJECTED_COMPOSED_CONTACT_FRESH_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during composed contact exam")
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
        "router-report",
        "v187-report",
        "v188-report",
        "map-report",
        "v205-dir",
        "v208-dir",
        "v209-report",
        "v212-report",
        "v213-report",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = exam(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.router_report,
        args.v187_report,
        args.v188_report,
        args.map_report,
        args.v205_dir,
        args.v208_dir,
        args.v209_report,
        args.v212_report,
        args.v213_report,
        args.output,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "development_score", "fresh_candidate_score", "report_hash")
            }
        )
    )


if __name__ == "__main__":
    main()
