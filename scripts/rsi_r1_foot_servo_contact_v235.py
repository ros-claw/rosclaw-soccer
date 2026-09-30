"""SIM_ONLY measured foot-ball servo development gate after coherent contact."""

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

from rosclaw_soccer.rsi.receiving_foot_servo_phase_motor import ReceivingFootServoPhaseMotor
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

SCHEMA = "rosclaw_soccer.rsi.r1_foot_servo_contact_v235.result.v1"
GAINS = ((0.2, 0.0), (0.35, 0.0), (0.0, 0.02), (0.2, 0.02), (0.35, 0.04), (0.5, 0.06))
ANCHOR = 158


def _run(task: tuple[Any, ...]) -> dict[str, Any]:
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
        position_gain,
        velocity_horizon,
    ) = task
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingFootServoPhaseMotor(
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
        servo_position_gain=position_gain,
        servo_velocity_horizon_sec=velocity_horizon,
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
        "servo_active_frames": feedback.servo_active_frames,
        "servo_peak_residual_rad": feedback.servo_peak_residual_rad,
        "servo_foot_index": feedback.servo_foot_index,
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


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    output = args.output
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY foot servo evidence required")
    v224, v229, v232, v233, v234 = (
        _checked(path)
        for path in (
            args.v224_report,
            args.v229_dir / "report.json",
            args.v232_dir / "report.json",
            args.v233_dir / "report.json",
            args.v234_dir / "report.json",
        )
    )
    if (
        v234["status"] != "REJECTED_COHERENT_LATENT_REFINEMENT_GATE"
        or v234["v233_report_hash"] != v233["report_hash"]
        or v233["v232_report_hash"] != v232["report_hash"]
        or v232["v229_report_hash"] != v229["report_hash"]
    ):
        raise ValueError("sealed contact latent failure lineage required")
    anchor = next(row for row in v233["candidates"] if row["source_episode"] == ANCHOR)
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
        raise ValueError("sealed precontact motor lineage required")
    pre_weights = replace(
        base_weights,
        output_bias=tuple(
            float(x) for x in np.asarray(base_weights.output_bias) + np.asarray(v224["best_offset"])
        ),
    )
    neural = _load_policy(args.v229_dir / "update-1.npz", v229["history"][0]["checkpoint_hash"])
    post_weights = replace(
        neural,
        output_bias=tuple(
            float(x) for x in np.asarray(neural.output_bias) + np.asarray(anchor["latent_offset"])
        ),
    )
    courses = tuple(ReceivingCourse(**row["course"]) for row in v232["course_summary"])
    if len(courses) != 11:
        raise ValueError("eleven consumed high-zone courses required")

    def tasks(gains: tuple[float, float]) -> list[tuple[Any, ...]]:
        return [
            (common[0], common[1], course, *common[2:], pre_weights, post_weights, *gains)
            for course in courses
        ]

    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_foot_servo_contact_v235.py",
            "src/rosclaw_soccer/rsi/receiving_foot_servo_phase_motor.py",
            "src/rosclaw_soccer/rsi/receiving_adaptive_phase_motor.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        baseline = list(pool.map(_run, tasks((0.0, 0.0))))
        if any(
            row["physical_trace_hash"] != sealed["physical_trace_hash"]
            for row, sealed in zip(baseline, anchor["rows"], strict=True)
        ):
            raise ValueError("zero foot servo must physically replay sealed coherent anchor")
        candidates = []
        base_wins = {
            row["course"]["seed"] for row in baseline if clean(row) and row["controlled_reception"]
        }
        for position_gain, velocity_horizon in GAINS:
            rows = list(pool.map(_run, tasks((position_gain, velocity_horizon))))
            wins = {
                row["course"]["seed"] for row in rows if clean(row) and row["controlled_reception"]
            }
            candidate = {
                "position_gain": position_gain,
                "velocity_horizon_sec": velocity_horizon,
                "score": _score(rows),
                "old_success_retained": base_wins <= wins,
                "rows": rows,
            }
            candidates.append(candidate)
            print(
                json.dumps(
                    {
                        "gains": [position_gain, velocity_horizon],
                        "score": candidate["score"][:2],
                        "retained": candidate["old_success_retained"],
                    }
                ),
                flush=True,
            )
    qualified = [
        row
        for row in candidates
        if row["old_success_retained"] and row["score"][0] >= 6 and row["score"][1] >= 9
    ]
    best = max(qualified or candidates, key=lambda row: tuple(row["score"]))
    report = {
        "schema": SCHEMA,
        "v234_report_hash": v234["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_ELEVEN_HIGH_ZONE_COURSES_ONLY",
        "zero_physical_equal": True,
        "baseline_score": _score(baseline),
        "baseline": baseline,
        "candidates": candidates,
        "best_gains": [best["position_gain"], best["velocity_horizon_sec"]],
        "best_score": best["score"],
        "status": "DEVELOPMENT_FOOT_SERVO_CANDIDATE_ONLY"
        if qualified
        else "REJECTED_FOOT_SERVO_CONTACT_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during foot servo test")
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
        "v224-report",
        "v229-dir",
        "v232-dir",
        "v233-dir",
        "v234-dir",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    report = evaluate(parser.parse_args())
    print(json.dumps({key: report[key] for key in ("status", "best_score", "report_hash")}))


if __name__ == "__main__":
    main()
