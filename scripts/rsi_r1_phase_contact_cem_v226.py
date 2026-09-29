"""SIM_ONLY measured-contact post-phase training on four neighboring courses."""

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
from rsi_r1_middle_basis_cem_v187 import PHYSICAL_KEYS
from rsi_r1_neighbor_contact_cem_v224 import rank

from rosclaw_soccer.rsi.receiving_phase_contact_motor import ReceivingPhaseContactMotor
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

SCHEMA = "rosclaw_soccer.rsi.r1_phase_contact_cem_v226.result.v1"
SEED = 226929
GENERATIONS = 2
POPULATION = 16


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
        weights,
        post_logits,
    ) = task
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = ReceivingPhaseContactMotor(
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


def train(
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
    v224_report_path: Path,
    v225_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY phase contact evidence required")
    v224, v225 = (_checked(path) for path in (v224_report_path, v225_report_path))
    if (
        v225["status"] != "REJECTED_NEIGHBOR_REFINEMENT_GATE"
        or v225["v224_report_hash"] != v224["report_hash"]
        or v224["best_rank"][0] != 2
        or v224["best_rank"][1] != 4
    ):
        raise ValueError("sealed clean near-boundary precontact skill required")
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
        raise ValueError("sealed motor contact lineage required")
    weights = replace(
        base_weights,
        output_bias=tuple(
            float(value)
            for value in np.asarray(base_weights.output_bias) + np.asarray(v224["best_offset"])
        ),
    )
    courses = tuple(ReceivingCourse(**row["course"]) for row in v224["best_rows"])

    def tasks(post_logits: tuple[float, ...]) -> list[tuple[Any, ...]]:
        return [
            (common[0], common[1], course, *common[2:], weights, post_logits) for course in courses
        ]

    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_phase_contact_cem_v226.py",
            "scripts/rsi_r1_contact_manifold_v222.py",
            "src/rosclaw_soccer/rsi/receiving_phase_contact_motor.py",
            "src/rosclaw_soccer/rsi/receiving_protected_composed_neural.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rng = np.random.default_rng(SEED)
    center = np.zeros(12, dtype=np.float64)
    sigma = 0.35
    history = []
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        zero = list(pool.map(_run_task, tasks((0.0,) * 12)))
        if any(
            row["physical_trace_hash"] != old["physical_trace_hash"]
            for row, old in zip(zero, v224["best_rows"], strict=True)
        ):
            raise ValueError("zero post-contact residual must physically replay clean parent")
        for generation in range(GENERATIONS):
            offsets = [center.copy()]
            offsets.extend(
                np.clip(center + rng.normal(0.0, sigma, 12), -1.2, 1.2)
                for _ in range(POPULATION - 1)
            )
            flat = list(
                pool.map(
                    _run_task,
                    [
                        task
                        for offset in offsets
                        for task in tasks(tuple(float(value) for value in offset))
                    ],
                )
            )
            population = [
                {
                    "post_logits": [float(value) for value in offsets[index]],
                    "rows": flat[index * 4 : index * 4 + 4],
                    "rank": rank(flat[index * 4 : index * 4 + 4]),
                }
                for index in range(POPULATION)
            ]
            population.sort(key=lambda row: tuple(row["rank"]), reverse=True)
            center = np.mean([row["post_logits"] for row in population[:4]], axis=0)
            sigma = max(0.08, sigma * 0.65)
            history.append({"generation": generation + 1, "population": population})
            print(
                json.dumps({"generation": generation + 1, "best": population[0]["rank"]}),
                flush=True,
            )
    best = max(
        (row for generation in history for row in generation["population"]),
        key=lambda row: tuple(row["rank"]),
    )
    qualified = (
        best["rank"][0] >= 3
        and best["rank"][1] == 4
        and all(row["post_active_frames"] > 0 for row in best["rows"])
    )
    report = {
        "schema": SCHEMA,
        "v225_report_hash": v225["report_hash"],
        "v224_report_hash": v224["report_hash"],
        **lineage,
        "source_hashes": sources,
        "partition": "CONSUMED_FOUR_COURSE_MEASURED_CONTACT_POST_PHASE_CEM",
        "seed": SEED,
        "course_seeds": v224["course_seeds"],
        "zero_post_replay": zero,
        "history": history,
        "best_post_logits": best["post_logits"],
        "best_rank": best["rank"],
        "best_rows": best["rows"],
        "status": "DEVELOPMENT_PHASE_CONTACT_CANDIDATE_ONLY"
        if qualified
        else "REJECTED_PHASE_CONTACT_CEM_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during post-contact phase training")
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
        "v225-report",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = train(
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
        args.v224_report,
        args.v225_report,
        args.output,
    )
    print(json.dumps({key: report[key] for key in ("status", "best_rank", "report_hash")}))


if __name__ == "__main__":
    main()
