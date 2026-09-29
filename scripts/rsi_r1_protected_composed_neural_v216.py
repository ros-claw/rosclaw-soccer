"""SIM_ONLY measured hard-retention replay of an online neural receiving candidate."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_composed_contact_fresh_v214 import FRESH_COURSES as CONSUMED_V214_COURSES
from rsi_r1_measured_skill_router_v207 import FRESH_COURSES as CONSUMED_V207_COURSES
from rsi_r1_measured_skill_router_v207 import _score
from rsi_r1_middle_basis_cem_v187 import PHYSICAL_KEYS
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy
from rsi_r1_temporal_zero_v192 import TRAIN_COURSES

from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.rsi.receiving_protected_composed_neural import (
    ReceivingProtectedComposedNeural,
)
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.rsi.team_receive_side_navigation import TeamReceiveSideNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window

SCHEMA = "rosclaw_soccer.rsi.r1_protected_composed_neural_v216.result.v1"


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
        weights,
        protected,
    ) = task
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingProtectedComposedNeural(
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


def verify(
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
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY protected neural evidence required")
    parent, right_parent, refine, lateral, v205, mapping, v214, v215 = (
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
        )
    )
    if (
        v215["status"] != "REJECTED_COMPOSED_NEURAL_PPO_GATE"
        or v215["v214_report_hash"] != v214["report_hash"]
        or not v215["baseline_physical_equal"]
        or v215["baseline_score"][:2] != [6, 14]
        or v215["best_update"] != 3
        or v205["map_report_hash"] != mapping["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed failed online neural lineage required")
    history = next(row for row in v215["history"] if row["update"] == v215["best_update"])
    weights = _load_policy(v215_dir / "update-3.npz", history["checkpoint_hash"])
    consumed = (*CONSUMED_V207_COURSES, *CONSUMED_V214_COURSES)
    parent_rows = (*v214["development"], *v214["fresh_candidate"])
    protected_seeds = tuple(
        row["course"]["seed"]
        for row in parent_rows
        if row["safe"] and not row["own_nonfoot_frames"] and row["controlled_reception"]
    )
    if len(protected_seeds) != 6:
        raise ValueError("exactly six genuine frozen successful skill courses required")
    neural_rows = {row["course"]["seed"]: row for row in history["deterministic_rows"]}
    protected = tuple(
        tuple(float(value) for value in neural_rows[seed]["observed_features"][0][:10])
        for seed in protected_seeds
    )
    failing_initial = [
        np.asarray(row["observed_features"][0][:10])
        for row in history["deterministic_rows"]
        if row["course"]["seed"] not in protected_seeds and row["observed_features"]
    ]
    minimum_failure_distance = min(
        float(np.linalg.norm(np.asarray(good) - bad))
        for good in protected
        for bad in failing_initial
    )
    if minimum_failure_distance <= 0.005:
        raise ValueError("measured protection radius overlaps a failed development condition")
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
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
    slope = tuple(lateral["best"]["slope"])
    low_weights = tuple(float(value) for value in v214["low_weights"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_protected_composed_neural_v216.py",
            "src/rosclaw_soccer/rsi/receiving_protected_composed_neural.py",
            "src/rosclaw_soccer/rsi/receiving_composed_neural_residual.py",
            "src/rosclaw_soccer/rsi/receiving_composed_contact_router.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)

    def tasks(courses: tuple[Any, ...]) -> list[tuple[Any, ...]]:
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
                weights,
                protected,
            )
            for course in courses
        ]

    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        rows = list(pool.map(_run_task, tasks(consumed)))
        anchors = list(pool.map(_run_task, tasks(TRAIN_COURSES[-2:])))
    protected_equal = all(
        row["protected_episode"] is True
        and row["physical_trace_hash"] == old["physical_trace_hash"]
        for row, old in zip(rows, parent_rows, strict=True)
        if row["course"]["seed"] in protected_seeds
    )
    anchor_equal = all(
        row["physical_trace_hash"] == old["physical_trace_hash"]
        for row, old in zip(anchors, v214["anchors"], strict=True)
    )
    score = _score(rows)
    report = {
        "schema": SCHEMA,
        "v215_report_hash": v215["report_hash"],
        "v214_report_hash": v214["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_SIX_SKILL_HARD_RETENTION_REPLAY_NOT_FRESH",
        "protected_seeds": protected_seeds,
        "protected_initial_features": protected,
        "minimum_failure_distance": minimum_failure_distance,
        "protected_physical_equal": protected_equal,
        "anchor_physical_equal": anchor_equal,
        "rows": rows,
        "score": score,
        "status": "DEVELOPMENT_PROTECTED_NEURAL_CANDIDATE_ONLY"
        if protected_equal and anchor_equal and score[0] > 6 and score[1] >= 14
        else "REJECTED_PROTECTED_NEURAL_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during protected neural replay")
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
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = verify(
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
        args.output,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "score", "protected_physical_equal", "report_hash")
            }
        )
    )


if __name__ == "__main__":
    main()
