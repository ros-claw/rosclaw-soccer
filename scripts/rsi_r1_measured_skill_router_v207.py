"""SIM_ONLY state-bound specialist routing and held-out receiving examination."""

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
from rsi_r1_middle_basis_cem_v187 import PHYSICAL_KEYS
from rsi_r1_temporal_actor_critic_v193 import clean
from rsi_r1_temporal_self_imitation_v196 import FRESH_COURSES as CONSUMED_V202_COURSES
from rsi_r1_temporal_zero_v192 import TRAIN_COURSES

from rosclaw_soccer.rsi.receiving_measured_skill_router import (
    ReceivingMeasuredSkillRouter,
    ReceivingSkillKnot,
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

SCHEMA = "rosclaw_soccer.rsi.r1_measured_skill_router_v207.result.v1"
FRESH_COURSES = tuple(
    ReceivingCourse("red.finisher", 207001 + i * 4 + j, speed, lateral)
    for i, lateral in enumerate((0.066, 0.072))
    for j, speed in enumerate((1.18, 1.25, 1.31, 1.34))
)


def _run_task(task: tuple[Any, ...]) -> dict[str, Any]:
    asset_root, policy_path, course, coordination, left, right, slope, knots = task
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingMeasuredSkillRouter(
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
        "measured_lateral_m": feedback.measured_lateral_m,
        "frames_recorded": n,
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


def _score(rows: list[dict[str, Any]]) -> tuple[int, int, float]:
    return (
        sum(clean(row) and row["controlled_reception"] for row in rows),
        sum(clean(row) for row in rows),
        -sum(
            row["tail_maximum_foot_distance_m"] + row["tail_maximum_ball_speed_mps"] for row in rows
        ),
    )


def exam(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    v187_report: Path,
    v188_report: Path,
    map_report: Path,
    old_zero_report: Path,
    v205_report: Path,
    v206_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY measured skill exam required")
    development_hash = preflight_receiving_courses(TRAIN_COURSES)
    consumed_hash = preflight_receiving_courses(CONSUMED_V202_COURSES)
    fresh_hash = preflight_receiving_courses(FRESH_COURSES)
    parent, right_parent, refine, lateral, v187, v188, mapping, zero, v205, v206 = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            v187_report,
            v188_report,
            map_report,
            old_zero_report,
            v205_report,
            v206_report,
        )
    )
    if (
        v205["status"] != "PHYSICALLY_VERIFIED_48D_TEACHERS_ONLY"
        or v206["status"] != "REJECTED_GENUINE_TEACHER_DISTILL_GATE"
        or v206["teacher_report_hash"] != v205["report_hash"]
        or zero["course_bank_hash"] != development_hash
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed genuine specialist and failed distillation lineage required")
    specialist_weights = {
        "parent": (0.0,) * 12,
        "low": tuple(_find_candidate(v188, 1, 6)["middle_weights"]),
        "high": tuple(_find_candidate(v187, 2, 2)["middle_weights"]),
    }
    first_features = {
        row["course"]["seed"]: row["observed_features"][0] for row in zero["rows"][:16]
    }
    best_by_seed: dict[int, dict[str, Any]] = {}
    for row in mapping["rows"]:
        summary = row["summary"]
        if (
            row["expert"] not in specialist_weights
            or not clean(summary)
            or not summary["controlled_reception"]
        ):
            continue
        seed = summary["course"]["seed"]
        old = best_by_seed.get(seed)
        if old is None or (
            summary["tail_maximum_foot_distance_m"] + summary["tail_maximum_ball_speed_mps"]
            < old["summary"]["tail_maximum_foot_distance_m"]
            + old["summary"]["tail_maximum_ball_speed_mps"]
        ):
            best_by_seed[seed] = row
    if len(best_by_seed) != 6:
        raise ValueError("six unique native specialist successes required")
    knots = tuple(
        ReceivingSkillKnot(
            float(first_features[seed][1] * 0.2),
            float(first_features[seed][3] * 2.0),
            row["expert"],
            tuple(float(value) for value in specialist_weights[row["expert"]]),
        )
        for seed, row in sorted(best_by_seed.items())
    )
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_measured_skill_router_v207.py",
            "src/rosclaw_soccer/rsi/receiving_measured_skill_router.py",
            "src/rosclaw_soccer/rsi/receiving_middle_basis_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:

        def evaluate(courses: tuple[ReceivingCourse, ...]) -> list[dict[str, Any]]:
            return list(
                pool.map(
                    _run_task,
                    [
                        (asset_root, policy_path, course, coordination, left, right, slope, knots)
                        for course in courses
                    ],
                )
            )

        development = evaluate(TRAIN_COURSES)
        retention_equal = all(
            row["selected_expert"] == "parent"
            and row["physical_trace_hash"] == old["physical_trace_hash"]
            for row, old in zip(development[-2:], zero["rows"][-2:], strict=True)
        )
        consumed = evaluate(CONSUMED_V202_COURSES)
        development_score = _score(development[:16])
        candidate_fresh = (
            evaluate(FRESH_COURSES) if retention_equal and development_score[0] >= 5 else []
        )
    qualified = retention_equal and len(candidate_fresh) == 8 and _score(candidate_fresh)[0] >= 6
    report = {
        "schema": SCHEMA,
        "v205_report_hash": v205["report_hash"],
        "v206_report_hash": v206["report_hash"],
        "source_hashes": sources,
        "partition": "MEASURED_SKILL_ROUTING_CONSUMED_THEN_NEW_FRESH",
        "development_bank_hash": development_hash,
        "consumed_v202_bank_hash": consumed_hash,
        "fresh_bank_hash": fresh_hash,
        "fresh_courses": [vars(course) for course in FRESH_COURSES],
        "skill_knots": [vars(knot) for knot in knots],
        "development": development,
        "development_score": development_score,
        "retention_physical_equal": retention_equal,
        "consumed_v202": consumed,
        "consumed_v202_score": _score(consumed),
        "fresh_candidate": candidate_fresh,
        "fresh_candidate_score": _score(candidate_fresh) if candidate_fresh else None,
        "status": "QUALIFIED_LOCAL_ROUTER_ONLY"
        if qualified
        else "REJECTED_MEASURED_SKILL_ROUTER_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during measured skill routing")
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
        "v187-report",
        "v188-report",
        "map-report",
        "old-zero-report",
        "v205-report",
        "v206-report",
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
        args.v187_report,
        args.v188_report,
        args.map_report,
        args.old_zero_report,
        args.v205_report,
        args.v206_report,
        args.output,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "development_score",
                    "consumed_v202_score",
                    "fresh_candidate_score",
                    "report_hash",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
