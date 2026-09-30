"""SIM_ONLY coherent 12D episode-latent contact exploration on hard courses."""

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
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy

from rosclaw_soccer.rsi.receiving_latent_phase_motor import ReceivingLatentPhaseMotor
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

SCHEMA = "rosclaw_soccer.rsi.r1_latent_contact_exploration_v232.result.v1"
SEED = 232929
LATENT_STD = 0.35
REPEATS = 16


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
        protected,
        pre_weights,
        post_weights,
        latent_seed,
        latent_std,
    ) = task
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingLatentPhaseMotor(
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
        latent_seed=latent_seed,
        latent_std=latent_std,
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
        "latent_seed": latent_seed,
        "latent_offset": feedback.latent_offset,
        "observed_frames": feedback.post_observed_frames,
        "observed_features": feedback.post_observed_features,
        "sampled_logits": feedback.post_sampled_logits,
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


def explore(
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
    v229_dir: Path,
    v231_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY latent contact evidence required")
    v223, v224, v229, v231 = (
        _checked(path)
        for path in (
            v223_report_path,
            v224_report_path,
            v229_dir / "report.json",
            v231_report_path,
        )
    )
    if (
        v231["status"] != "DEVELOPMENT_CONTACT_MODE_DIAGNOSIS_ONLY"
        or v231["v229_report_hash"] != v229["report_hash"]
        or v229["status"] != "REJECTED_ADAPTIVE_PHASE_PPO_GATE"
        or v224["v223_report_hash"] != v223["report_hash"]
    ):
        raise ValueError("sealed iid-noise failure and contact skills required")
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
        raise ValueError("sealed precontact motor lineage required")
    pre_weights = replace(
        base_weights,
        output_bias=tuple(
            float(value)
            for value in np.asarray(base_weights.output_bias) + np.asarray(v224["best_offset"])
        ),
    )
    post_weights = _load_policy(v229_dir / "update-1.npz", v229["history"][0]["checkpoint_hash"])
    courses = tuple(ReceivingCourse(**row["course"]) for row in v223["rows"]["baseline"])
    high_courses = tuple(
        course
        for course, row in zip(courses, v223["rows"]["baseline"], strict=True)
        if row["selected_expert"] == "high"
    )
    if len(high_courses) != 11:
        raise ValueError("eleven high-zone contact courses required")

    def tasks(
        course_bank: tuple[ReceivingCourse, ...], seeds: tuple[int, ...], std: float
    ) -> list[tuple[Any, ...]]:
        return [
            (common[0], common[1], course, *common[2:], pre_weights, post_weights, seed, std)
            for course, seed in zip(course_bank, seeds, strict=True)
        ]

    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_latent_contact_exploration_v232.py",
            "src/rosclaw_soccer/rsi/receiving_latent_phase_motor.py",
            "src/rosclaw_soccer/rsi/receiving_adaptive_phase_motor.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    sample_courses = high_courses * REPEATS
    seeds = tuple(int(value) for value in rng.integers(0, 2**31, len(sample_courses)))
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        zero = list(pool.map(_run_task, tasks(high_courses, (0,) * len(high_courses), 0.0)))
        old = {
            row["course"]["seed"]: row
            for row in v229["history"][0]["deterministic_rows"]
            if row["course"]["seed"] in {course.seed for course in high_courses}
        }
        if any(
            row["physical_trace_hash"] != old[row["course"]["seed"]]["physical_trace_hash"]
            for row in zero
        ):
            raise ValueError("zero episode latent must physically replay sealed neural actor")
        sampled = list(pool.map(_run_task, tasks(sample_courses, seeds, LATENT_STD)))
    records = []
    for index, row in enumerate(sampled):
        path = output / f"episode-{index}.npz"
        np.savez_compressed(
            path,
            frames=np.asarray(row["observed_frames"], dtype=np.int64),
            features=np.asarray(row["observed_features"], dtype=np.float64).reshape(-1, 48),
            logits=np.asarray(row["sampled_logits"], dtype=np.float64).reshape(-1, 12),
            latent=np.asarray(row["latent_offset"], dtype=np.float64),
        )
        records.append(
            {
                "index": index,
                "course": row["course"],
                "latent_seed": row["latent_seed"],
                "latent_offset": row["latent_offset"],
                "safe": row["safe"],
                "controlled_reception": row["controlled_reception"],
                "own_nonfoot_frames": row["own_nonfoot_frames"],
                "first_foot_frame": row["first_foot_frame"],
                "distance_m": row["tail_maximum_foot_distance_m"],
                "speed_mps": row["tail_maximum_ball_speed_mps"],
                "post_active_frames": row["post_active_frames"],
                "physical_trace_hash": row["physical_trace_hash"],
                "trajectory_hash": hash_bytes(path.read_bytes()),
            }
        )
    course_summary = []
    for course in high_courses:
        subset = [row for row in sampled if row["course"]["seed"] == course.seed]
        course_summary.append(
            {
                "course": vars(course),
                "clean_controlled": sum(
                    clean(row) and row["controlled_reception"] for row in subset
                ),
                "clean": sum(clean(row) for row in subset),
                "episodes": len(subset),
            }
        )
    report = {
        "schema": SCHEMA,
        "v231_report_hash": v231["report_hash"],
        "v229_report_hash": v229["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_HIGH_ZONE_COHERENT_LATENT_EXPLORATION_ONLY",
        "seed": SEED,
        "latent_std": LATENT_STD,
        "repeats": REPEATS,
        "zero_physical_equal": True,
        "zero_score": _score(zero),
        "sample_score": _score(sampled),
        "course_summary": course_summary,
        "episode_records": records,
        "status": "DEVELOPMENT_COHERENT_LATENT_EXPLORATION_ONLY",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during coherent latent exploration")
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
        "v231-report",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = explore(
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
        args.v229_dir,
        args.v231_report,
        args.output,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "zero_score", "sample_score", "course_summary", "report_hash")
            }
        )
    )


if __name__ == "__main__":
    main()
