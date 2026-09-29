"""SIM_ONLY safe state-gated phase skill, broad retention and local fresh exam."""

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
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_middle_basis_cem_v187 import PHYSICAL_KEYS
from rsi_r1_phase_contact_cem_v226 import _run_task as _run_ungated

from rosclaw_soccer.rsi.receiving_gated_phase_contact_motor import (
    ReceivingGatedPhaseContactMotor,
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

SCHEMA = "rosclaw_soccer.rsi.r1_gated_phase_fresh_v228.result.v1"
FRESH_COURSES = tuple(
    ReceivingCourse("red.finisher", 228001 + i * 4 + j, speed, lateral)
    for i, lateral in enumerate((0.068, 0.077))
    for j, speed in enumerate((1.22, 1.24, 1.26, 1.28))
)


def _run_gated(task: tuple[Any, ...]) -> dict[str, Any]:
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
        weights,
        post_logits,
        states,
    ) = task
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingGatedPhaseContactMotor(
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
        neural_weights=weights,
        protected_initial_features=protected,
        post_contact_logits=post_logits,
        activation_states=states,
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
        "activation_enabled": feedback.activation_enabled,
        "post_active_frames": feedback.post_active_frames,
        "post_peak_residual_rad": feedback.post_peak_residual_rad,
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


def exam(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    v205_dir: Path,
    map_report_path: Path,
    v214_report_path: Path,
    v215_dir: Path,
    v216_report_path: Path,
    v218_dir: Path,
    v220_report_path: Path,
    v223_report_path: Path,
    v224_report_path: Path,
    v226_report_path: Path,
    v227_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY gated phase exam evidence required")
    fresh_hash = preflight_receiving_courses(FRESH_COURSES)
    v223, v224, v226, v227 = (
        _checked(path)
        for path in (v223_report_path, v224_report_path, v226_report_path, v227_report_path)
    )
    if (
        v227["status"] != "REJECTED_PHASE_CONTACT_RETENTION_GATE"
        or v227["v226_report_hash"] != v226["report_hash"]
        or v226["v224_report_hash"] != v224["report_hash"]
        or v224["v223_report_hash"] != v223["report_hash"]
        or v226["best_rank"][:2] != [4, 4]
    ):
        raise ValueError("sealed phase-contact success and safety rejection required")
    common, (base_weights, _, _), lineage = context(
        asset_root,
        policy_path,
        parent_report,
        right_report,
        refine_report,
        lateral_report,
        v205_dir,
        map_report_path,
        v214_report_path,
        v215_dir,
        v216_report_path,
        v218_dir,
        v220_report_path,
    )
    if lineage["v220_report_hash"] != v224["v220_report_hash"]:
        raise ValueError("sealed contact motor lineage required")
    weights = replace(
        base_weights,
        output_bias=tuple(
            float(value)
            for value in np.asarray(base_weights.output_bias) + np.asarray(v224["best_offset"])
        ),
    )
    states = tuple(
        tuple(float(value) for value in row["observed_features"][0][:10])
        for row in v224["best_rows"]
        if row["course"]["seed"] in (223010, 223031)
    )
    if len(states) != 2:
        raise ValueError("two measured clean improvement states required")
    post = tuple(float(value) for value in v226["best_post_logits"])
    old_courses = tuple(ReceivingCourse(**row["course"]) for row in v223["rows"]["baseline"])

    def parent_tasks(courses: tuple[ReceivingCourse, ...]) -> list[tuple[Any, ...]]:
        return [
            (common[0], common[1], course, *common[2:], weights, (0.0,) * 12) for course in courses
        ]

    def candidate_tasks(courses: tuple[ReceivingCourse, ...]) -> list[tuple[Any, ...]]:
        return [
            (common[0], common[1], course, *common[2:], weights, post, states) for course in courses
        ]

    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_gated_phase_fresh_v228.py",
            "scripts/rsi_r1_phase_contact_cem_v226.py",
            "src/rosclaw_soccer/rsi/receiving_gated_phase_contact_motor.py",
            "src/rosclaw_soccer/rsi/receiving_phase_contact_motor.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        old_parent = list(pool.map(_run_ungated, parent_tasks(old_courses)))
        old_candidate = list(pool.map(_run_gated, candidate_tasks(old_courses)))
        parent_score = _score(old_parent)
        candidate_score = _score(old_candidate)
        unchanged = all(
            row["physical_trace_hash"] == old["physical_trace_hash"]
            for row, old in zip(old_candidate, old_parent, strict=True)
            if row["course"]["seed"] not in (223010, 223031)
        )
        retained = all(
            not (clean(old) and old["controlled_reception"])
            or (clean(row) and row["controlled_reception"])
            for old, row in zip(old_parent, old_candidate, strict=True)
        )
        eligible = (
            unchanged
            and retained
            and candidate_score[0] >= parent_score[0] + 2
            and candidate_score[1] >= parent_score[1]
        )
        print(
            json.dumps(
                {
                    "old_parent": parent_score,
                    "old_candidate": candidate_score,
                    "unchanged": unchanged,
                    "eligible": eligible,
                }
            ),
            flush=True,
        )
        if eligible:
            fresh_parent = list(pool.map(_run_ungated, parent_tasks(FRESH_COURSES)))
            fresh_candidate = list(pool.map(_run_gated, candidate_tasks(FRESH_COURSES)))
        else:
            fresh_parent = []
            fresh_candidate = []
    fresh_parent_score = _score(fresh_parent) if eligible else None
    fresh_candidate_score = _score(fresh_candidate) if eligible else None
    qualified = bool(
        eligible
        and fresh_parent_score is not None
        and fresh_candidate_score is not None
        and fresh_candidate_score[0] >= 6
        and fresh_candidate_score[1] >= 7
        and fresh_candidate_score[0] >= fresh_parent_score[0] + 2
    )
    report = {
        "schema": SCHEMA,
        "v227_report_hash": v227["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_GATED_RETENTION_THEN_PREDECLARED_LOCAL_FRESH8",
        "activation_states": states,
        "activation_radius": 0.014,
        "fresh_bank_hash": fresh_hash,
        "fresh_courses": [vars(course) for course in FRESH_COURSES],
        "old_parent": old_parent,
        "old_candidate": old_candidate,
        "old_parent_score": parent_score,
        "old_candidate_score": candidate_score,
        "old_unchanged_elsewhere": unchanged,
        "old_success_retained": retained,
        "fresh_parent": fresh_parent,
        "fresh_candidate": fresh_candidate,
        "fresh_parent_score": fresh_parent_score,
        "fresh_candidate_score": fresh_candidate_score,
        "status": (
            "FRESH_GATED_PHASE_QUALIFIED_FOR_NEXT_CHAIN_GATE"
            if qualified
            else "REJECTED_GATED_PHASE_FRESH_GATE"
            if eligible
            else "REJECTED_GATED_PHASE_RETENTION_GATE"
        ),
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during gated phase exam")
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
        "v226-report",
        "v227-report",
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
        args.v205_dir,
        args.map_report,
        args.v214_report,
        args.v215_dir,
        args.v216_report,
        args.v218_dir,
        args.v220_report,
        args.v223_report,
        args.v224_report,
        args.v226_report,
        args.v227_report,
        args.output,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "old_parent_score",
                    "old_candidate_score",
                    "fresh_parent_score",
                    "fresh_candidate_score",
                    "report_hash",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
