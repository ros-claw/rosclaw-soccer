"""SIM_ONLY replay of genuine high-speed specialist successes with 48D body states."""

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
from rsi_r1_course_map_v190 import COURSES, _find_candidate
from rsi_r1_middle_basis_cem_v187 import PHYSICAL_KEYS
from rsi_r1_temporal_actor_critic_v193 import clean

from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import ReceivingKinematicTemporalExpert
from rosclaw_soccer.rsi.receiving_middle_basis_expert import ReceivingMiddleBasisExpert
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.rsi.team_receive_side_navigation import TeamReceiveSideNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_kinematic_teacher_capture_v205.result.v1"


def _old_fraction(frame: int) -> float:
    if frame < 19:
        return max(0.0, (frame - 15) / 4.0)
    if frame <= 30:
        return 1.0
    if frame < 40:
        return (40 - frame) / 10.0
    return 0.0


def _new_fraction(frame: int) -> float:
    if frame < 19:
        return max(0.0, (frame - 15) / 4.0)
    if frame <= 50:
        return 1.0
    return max(0.0, (65 - frame) / 15.0)


@dataclass
class RecordingMiddleExpert(ReceivingMiddleBasisExpert):
    requires_shin_clearance: bool = field(init=False, default=True)
    observed_frames: list[int] = field(init=False, default_factory=list)
    observed_features: list[tuple[float, ...]] = field(init=False, default_factory=list)
    nominal_target_actions: list[tuple[float, ...]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.recording_middle_teacher.v1",
                "parent_contract_hash": self.contract_hash,
                "requires_shin_clearance": True,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        target = super().propose(observation)
        if 15 <= observation.frame < 65:
            features = ReceivingKinematicTemporalExpert.features(observation)
            fraction = _old_fraction(observation.frame) / max(
                _new_fraction(observation.frame), 1e-6
            )
            nominal = np.clip(
                fraction * float(self.middle_feature or 0.0) * np.asarray(self.middle_weights),
                -0.999,
                0.999,
            )
            self.observed_frames.append(observation.frame)
            self.observed_features.append(features)
            self.nominal_target_actions.append(tuple(float(value) for value in nominal))
        return target


def _run_task(task: tuple[Any, ...]) -> dict[str, Any]:
    asset_root, policy_path, course, coordination, left, right, slope, weights = task
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    feedback = RecordingMiddleExpert(
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
        middle_weights=weights,
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
    _, outcome = receiving_window(
        trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
    )
    return {
        "course": vars(course),
        "measured_lateral_m": feedback.measured_lateral_m,
        "middle_feature": feedback.middle_feature,
        "safe": info["safe"],
        "controlled_reception": outcome["controlled_reception"],
        "physical_trace_hash": hash_json(
            {key: np.asarray(trace[key]).tolist() for key in PHYSICAL_KEYS}
        ),
        "frames": feedback.observed_frames,
        "features": feedback.observed_features,
        "nominal_target_actions": feedback.nominal_target_actions,
    }


def capture(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    v187_report: Path,
    v188_report: Path,
    map_report: Path,
    v204_report: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY specialist teacher evidence required")
    bank_hash = preflight_receiving_courses(COURSES)
    parent, right_parent, refine, lateral, v187, v188, mapping, v204 = (
        _checked(path)
        for path in (
            parent_report,
            right_report,
            refine_report,
            lateral_report,
            v187_report,
            v188_report,
            map_report,
            v204_report,
        )
    )
    if (
        mapping["course_bank_hash"] != bank_hash
        or v204["status"] != "REJECTED_ONLINE_KINEMATIC_PPO_GATE"
        or parent["policy_hash"] != hash_bytes(policy_path.read_bytes())
    ):
        raise ValueError("sealed failed PPO and genuine specialist map required")
    specialists = {
        "parent": (0.0,) * 12,
        "low": tuple(_find_candidate(v188, 1, 6)["middle_weights"]),
        "high": tuple(_find_candidate(v187, 2, 2)["middle_weights"]),
    }
    successes = [
        row
        for row in mapping["rows"]
        if row["expert"] in specialists
        and clean(row["summary"])
        and row["summary"]["controlled_reception"]
    ]
    if (
        len(successes) != 7
        or sum(row["summary"]["course"]["speed_mps"] >= 1.28 for row in successes) != 2
    ):
        raise ValueError(
            "seven sealed physical teacher successes including two high-speed required"
        )
    coordination = tuple(parent["selected"]["weights"])
    left = tuple(refine["best"]["left_weights"])
    right = tuple(float(value) for value in np.asarray(right_parent["best_right"]["weights"]) * 0.7)
    slope = tuple(lateral["best"]["slope"])
    tasks = [
        (
            asset_root,
            policy_path,
            ReceivingCourse(**row["summary"]["course"]),
            coordination,
            left,
            right,
            slope,
            specialists[row["expert"]],
        )
        for row in successes
    ]
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_kinematic_teacher_capture_v205.py",
            "src/rosclaw_soccer/rsi/receiving_middle_basis_expert.py",
            "src/rosclaw_soccer/rsi/receiving_kinematic_temporal_expert.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    rows = []
    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for index, episode in enumerate(pool.map(_run_task, tasks)):
            teacher = successes[index]
            frames = np.asarray(episode.pop("frames"), dtype=np.int64)
            features = np.asarray(episode.pop("features"), dtype=np.float64)
            nominal = np.asarray(episode.pop("nominal_target_actions"), dtype=np.float64)
            equal = episode["physical_trace_hash"] == teacher["summary"]["physical_trace_hash"]
            if (
                not equal
                or not episode["safe"]
                or not episode["controlled_reception"]
                or not np.array_equal(frames, np.arange(15, 65))
                or features.shape != (50, 48)
                or nominal.shape != (50, 12)
                or not np.isfinite(features).all()
                or not np.isfinite(nominal).all()
            ):
                raise ValueError("genuine physically equal 48D teacher success required")
            path = output / f"teacher-{index}.npz"
            np.savez_compressed(path, frames=frames, features=features, target_actions=nominal)
            rows.append(
                {
                    "expert": teacher["expert"],
                    "course": episode["course"],
                    "physical_trace_hash": episode["physical_trace_hash"],
                    "historical_physical_trace_hash": teacher["summary"]["physical_trace_hash"],
                    "physical_equal": equal,
                    "feature_hash": hash_bytes(path.read_bytes()),
                    "middle_feature": episode["middle_feature"],
                }
            )
            print(
                json.dumps({"teacher": index, "expert": teacher["expert"], "equal": equal}),
                flush=True,
            )
    report = {
        "schema": SCHEMA,
        "map_report_hash": mapping["report_hash"],
        "v204_report_hash": v204["report_hash"],
        "source_hashes": sources,
        "partition": "REPLAY_CONSUMED_GENUINE_TEACHER_SUCCESSES_NOT_FRESH",
        "course_bank_hash": bank_hash,
        "rows": rows,
        "physical_equal_count": sum(row["physical_equal"] for row in rows),
        "high_speed_teacher_count": sum(row["course"]["speed_mps"] >= 1.28 for row in rows),
        "status": "PHYSICALLY_VERIFIED_48D_TEACHERS_ONLY",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during teacher capture")
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
        "v204-report",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = capture(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.v187_report,
        args.v188_report,
        args.map_report,
        args.v204_report,
        args.output,
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "physical_equal_count",
                    "high_speed_teacher_count",
                    "report_hash",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
