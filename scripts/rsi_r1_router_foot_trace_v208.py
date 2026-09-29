"""SIM_ONLY measured-foot replay of failed router holdout, for contact-phase diagnosis."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_measured_skill_router_v207 import FRESH_COURSES
from rsi_r1_middle_basis_cem_v187 import PHYSICAL_KEYS

from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import ReceivingKinematicTemporalExpert
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
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule

SCHEMA = "rosclaw_soccer.rsi.r1_router_foot_trace_v208.result.v1"


@dataclass
class RecordingRouter(ReceivingMeasuredSkillRouter):
    requires_shin_clearance: bool = field(init=False, default=True)
    observed_frames: list[int] = field(init=False, default_factory=list)
    observed_features: list[tuple[float, ...]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.recording_router_foot_trace.v1",
                "parent_contract_hash": self.contract_hash,
                "requires_shin_clearance": True,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        target = super().propose(observation)
        if 15 <= observation.frame < 65:
            self.observed_frames.append(observation.frame)
            self.observed_features.append(ReceivingKinematicTemporalExpert.features(observation))
        return target


def _run_task(task: tuple[Any, ...]) -> dict[str, Any]:
    asset_root, policy_path, course, coordination, left, right, slope, knots = task
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = RecordingRouter(
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
    return {
        "course": vars(course),
        "selected_expert": feedback.selected_expert,
        "physical_trace_hash": hash_json(
            {key: np.asarray(trace[key]).tolist() for key in PHYSICAL_KEYS}
        ),
        "safe": result.to_dict()["safe"],
        "frames": feedback.observed_frames,
        "features": feedback.observed_features,
    }


def diagnose(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    router_report_path: Path,
    teacher_dir: Path,
    map_report_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY failed-router foot trace required")
    parent, right_parent, refine, lateral, router, teachers, mapping = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            router_report_path,
            teacher_dir / "report.json",
            map_report_path,
        )
    )
    if (
        router["status"] != "REJECTED_MEASURED_SKILL_ROUTER_GATE"
        or len(router["fresh_candidate"]) != 8
        or router["fresh_candidate_score"][0] != 0
        or teachers["physical_equal_count"] != 7
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed eight failed fresh cases and verified teachers required")
    knots = tuple(
        ReceivingSkillKnot(
            float(row["lateral_m"]),
            float(row["closing_speed_mps"]),
            row["expert"],
            tuple(float(value) for value in row["weights"]),
        )
        for row in router["skill_knots"]
    )
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
    slope = tuple(lateral["best"]["slope"])
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_router_foot_trace_v208.py",
            "src/rosclaw_soccer/rsi/receiving_measured_skill_router.py",
            "src/rosclaw_soccer/rsi/receiving_kinematic_temporal_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    tasks = [
        (asset_root, policy_path, course, coordination, left, right, slope, knots)
        for course in FRESH_COURSES
    ]
    output.mkdir(parents=True)
    rows = []
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for index, episode in enumerate(pool.map(_run_task, tasks)):
            old = router["fresh_candidate"][index]
            features = np.asarray(episode.pop("features"), dtype=np.float64)
            frames = np.asarray(episode.pop("frames"), dtype=np.int64)
            if (
                episode["physical_trace_hash"] != old["physical_trace_hash"]
                or episode["selected_expert"] != old["selected_expert"]
                or not np.array_equal(frames, np.arange(15, 65))
                or features.shape != (50, 48)
                or not np.isfinite(features).all()
            ):
                raise ValueError("failed-router 48D replay must be physically equal")
            path = output / f"fresh-{old['course']['seed']}.npz"
            np.savez_compressed(path, frames=frames, features=features)
            candidates = []
            for teacher_index, teacher in enumerate(teachers["rows"]):
                if teacher["expert"] != episode["selected_expert"]:
                    continue
                teacher_path = teacher_dir / f"teacher-{teacher_index}.npz"
                if hash_bytes(teacher_path.read_bytes()) != teacher["feature_hash"]:
                    raise ValueError("sealed teacher foot trace required")
                with np.load(teacher_path, allow_pickle=False) as arrays:
                    teacher_features = np.asarray(arrays["features"], dtype=np.float64)
                measured_distance = float(
                    np.hypot(
                        (features[0, 1] - teacher_features[0, 1]) * 0.2 / 0.006,
                        (features[0, 3] - teacher_features[0, 3]) * 2.0 / 0.12,
                    )
                )
                candidates.append((measured_distance, teacher_index, teacher, teacher_features))
            _, teacher_index, teacher, teacher_features = min(candidates, key=lambda row: row[0])
            teacher_history = next(
                row["summary"]
                for row in mapping["rows"]
                if row["expert"] == teacher["expert"]
                and row["summary"]["course"]["seed"] == teacher["course"]["seed"]
            )
            fresh_touch = int(old["first_foot_frame"])
            teacher_touch = int(teacher_history["first_foot_frame"])
            phase_errors = {}
            for offset in (-5, -2, 0, 2, 5):
                fi = fresh_touch + offset - 15
                ti = teacher_touch + offset - 15
                if 0 <= fi < 50 and 0 <= ti < 50:
                    fresh_feet = features[fi, 10:16].reshape(2, 3) * 0.5
                    teacher_feet = teacher_features[ti, 10:16].reshape(2, 3) * 0.5
                    phase_errors[str(offset)] = [
                        float(value) for value in np.linalg.norm(fresh_feet - teacher_feet, axis=1)
                    ]
            rows.append(
                {
                    "course": old["course"],
                    "selected_expert": old["selected_expert"],
                    "physical_trace_hash": old["physical_trace_hash"],
                    "feature_hash": hash_bytes(path.read_bytes()),
                    "teacher_index": teacher_index,
                    "teacher_course": teacher["course"],
                    "fresh_first_foot_frame": fresh_touch,
                    "teacher_first_foot_frame": teacher_touch,
                    "phase_foot_relative_errors_m": phase_errors,
                    "fresh_tail_distance_m": old["tail_maximum_foot_distance_m"],
                    "fresh_tail_speed_mps": old["tail_maximum_ball_speed_mps"],
                    "fresh_nonfoot_frames": old["own_nonfoot_frames"],
                }
            )
            print(json.dumps({"case": index, "phase_errors_m": phase_errors}), flush=True)
    report = {
        "schema": SCHEMA,
        "router_report_hash": router["report_hash"],
        "teacher_report_hash": teachers["report_hash"],
        "source_hashes": sources,
        "partition": "READ_ONLY_CONSUMED_FRESH_FOOT_PHASE_DIAGNOSTIC",
        "rows": rows,
        "physical_equal_count": len(rows),
        "status": "CONSUMED_FOOT_PHASE_DIAGNOSTIC_ONLY",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during failed-router foot diagnosis")
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
        "teacher-dir",
        "map-report",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.router_report,
        args.teacher_dir,
        args.map_report,
        args.output,
    )
    print(
        json.dumps({key: report[key] for key in ("status", "physical_equal_count", "report_hash")})
    )


if __name__ == "__main__":
    main()
