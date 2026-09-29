"""SIM_ONLY causal measured-state skill recall and predeclared fresh exam."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_composed_contact_fresh_v214 import FRESH_COURSES as V214_COURSES
from rsi_r1_measured_skill_router_v207 import FRESH_COURSES as V207_COURSES
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_middle_basis_cem_v187 import PHYSICAL_KEYS
from rsi_r1_protected_composed_neural_v216 import _run_task as _run_baseline
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy
from rsi_r1_protected_neural_fresh_v217 import FRESH_COURSES as V217_COURSES

from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_growing_skill_bank import ReceivingGrowingSkillBank
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

SCHEMA = "rosclaw_soccer.rsi.r1_growing_skill_bank_v219.result.v1"
FRESH_COURSES = tuple(
    ReceivingCourse("red.finisher", 219001 + i * 4 + j, speed, lateral)
    for i, lateral in enumerate((0.068, 0.072))
    for j, speed in enumerate((1.15, 1.19, 1.26, 1.34))
)


def _run_candidate(task: tuple[Any, ...]) -> dict[str, Any]:
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
        baseline_weights,
        protected,
        learned_weights,
        learned_initial,
    ) = task
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingGrowingSkillBank(
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
        neural_weights=baseline_weights,
        protected_initial_features=protected,
        learned_weights=learned_weights,
        learned_initial_features=learned_initial,
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
        "learned_episode": feedback.learned_episode,
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
    v217_report_path: Path,
    v218_dir: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY growing skill-bank evidence required")
    fresh_hash = preflight_receiving_courses(FRESH_COURSES)
    parent, right_parent, refine, lateral, v205, mapping, v214, v215, v216, v217, v218 = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            v205_dir / "report.json",
            map_report_path,
            v214_report_path,
            v215_dir / "report.json",
            v216_report_path,
            v217_report_path,
            v218_dir / "report.json",
        )
    )
    if (
        v218["status"] != "REJECTED_PROTECTED_ONLINE_PPO_GATE"
        or v218["v217_report_hash"] != v217["report_hash"]
        or v217["v216_report_hash"] != v216["report_hash"]
        or v216["v215_report_hash"] != v215["report_hash"]
        or v218["baseline_score"][:2] != [11, 23]
        or v205["map_report_hash"] != mapping["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed failed online learning lineage required")
    baseline_best = next(row for row in v215["history"] if row["update"] == 3)
    baseline_weights = _load_policy(v215_dir / "update-3.npz", baseline_best["checkpoint_hash"])
    learned_best = next(row for row in v218["history"] if row["update"] == 3)
    learned_weights = _load_policy(v218_dir / "update-3.npz", learned_best["checkpoint_hash"])
    learned_row = next(
        row for row in learned_best["deterministic_rows"] if row["course"]["seed"] == 217005
    )
    if not clean(learned_row) or not learned_row["controlled_reception"]:
        raise ValueError("genuinely learned successful contact required")
    learned_initial = tuple(float(value) for value in learned_row["observed_features"][0][:10])
    protected = tuple(
        tuple(float(value) for value in row) for row in v216["protected_initial_features"]
    )
    other_states = [
        np.asarray(row["observed_features"][0][:10])
        for row in learned_best["deterministic_rows"]
        if row["course"]["seed"] != 217005 and row["observed_features"]
    ]
    nearest_other = min(
        float(np.linalg.norm(np.asarray(learned_initial) - state)) for state in other_states
    )
    if nearest_other <= 0.02:
        raise ValueError("new skill recall overlaps known consumed states")
    knots = tuple(
        ReceivingSkillKnot(
            float(row["lateral_m"]),
            float(row["closing_speed_mps"]),
            row["expert"],
            tuple(float(value) for value in row["weights"]),
        )
        for row in v214["knots"]
    )
    references = []
    for index, row in enumerate(v205["rows"]):
        if row["expert"] != "high":
            continue
        path = v205_dir / f"teacher-{index}.npz"
        if hash_bytes(path.read_bytes()) != row["feature_hash"]:
            raise ValueError("sealed physical high teacher required")
        with np.load(path, allow_pickle=False) as arrays:
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
    common = (
        asset_root,
        policy_path,
        tuple(parent["selected"]["weights"]),
        tuple(refine["best"]["left_weights"]),
        tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7),
        tuple(lateral["best"]["slope"]),
        knots,
        tuple(references),
        tuple(float(value) for value in v214["low_weights"]),
        baseline_weights,
        protected,
    )

    def tasks(courses: tuple[ReceivingCourse, ...], candidate: bool) -> list[tuple[Any, ...]]:
        return [
            (
                common[0],
                common[1],
                course,
                *common[2:],
                *((learned_weights, learned_initial) if candidate else ()),
            )
            for course in courses
        ]

    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_growing_skill_bank_v219.py",
            "src/rosclaw_soccer/rsi/receiving_growing_skill_bank.py",
            "src/rosclaw_soccer/rsi/receiving_protected_composed_neural.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    consumed = (*V207_COURSES, *V214_COURSES, *V217_COURSES)
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        development = list(pool.map(_run_candidate, tasks(consumed, True)))
        baseline_rows = (*v216["rows"], *v217["candidate"])
        preserved = all(
            row["physical_trace_hash"] == old["physical_trace_hash"]
            for row, old in zip(development, baseline_rows, strict=True)
            if row["course"]["seed"] != 217005
        )
        learned = next(row for row in development if row["course"]["seed"] == 217005)
        development_score = _score(development)
        eligible = (
            preserved
            and learned["learned_episode"]
            and clean(learned)
            and learned["controlled_reception"]
            and development_score[0] >= 12
            and development_score[1] >= 23
        )
        print(
            json.dumps(
                {"development": development_score, "preserved": preserved, "eligible": eligible}
            ),
            flush=True,
        )
        if eligible:
            fresh_baseline = list(pool.map(_run_baseline, tasks(FRESH_COURSES, False)))
            fresh_candidate = list(pool.map(_run_candidate, tasks(FRESH_COURSES, True)))
        else:
            fresh_baseline = []
            fresh_candidate = []
    baseline_score = _score(fresh_baseline) if eligible else None
    candidate_score = _score(fresh_candidate) if eligible else None
    fresh_qualified = bool(
        eligible
        and candidate_score is not None
        and baseline_score is not None
        and candidate_score[0] >= 6
        and candidate_score[1] >= 7
        and candidate_score[0] >= baseline_score[0] + 2
    )
    report = {
        "schema": SCHEMA,
        "v218_report_hash": v218["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_SKILL_BANK_DEVELOPMENT_THEN_PREDECLARED_FRESH8",
        "nearest_consumed_state_distance": nearest_other,
        "learned_initial_features": learned_initial,
        "fresh_bank_hash": fresh_hash,
        "fresh_courses": [vars(course) for course in FRESH_COURSES],
        "development": development,
        "development_score": development_score,
        "preserved_physical_equal": preserved,
        "fresh_parent": fresh_baseline,
        "fresh_candidate": fresh_candidate,
        "fresh_parent_score": baseline_score,
        "fresh_candidate_score": candidate_score,
        "status": (
            "FRESH_GROWING_SKILL_BANK_QUALIFIED_FOR_NEXT_CHAIN_GATE"
            if fresh_qualified
            else "REJECTED_GROWING_SKILL_BANK_FRESH_GATE"
            if eligible
            else "REJECTED_GROWING_SKILL_BANK_DEVELOPMENT_GATE"
        ),
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during growing skill bank exam")
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
        "v217-report",
        "v218-dir",
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
        args.v217_report,
        args.v218_dir,
        args.output,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "development_score",
                    "fresh_parent_score",
                    "fresh_candidate_score",
                    "report_hash",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
