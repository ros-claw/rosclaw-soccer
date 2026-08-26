"""Failure-driven reset curriculum for generic post-skill body recovery."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.goalkeeper_v2.motion_library import (
    GoalkeeperMotionFamily,
    load_goalkeeper_motion_library,
)
from rosclaw_soccer.training.recovery_snapshot import (
    RecoverySnapshot,
    load_recovery_snapshot_corpus,
)


@dataclass(frozen=True)
class RecoveryReplayConfig:
    """Sampling policy for stability-preserving failure replay."""

    failure_weight: float = 3.0
    high_momentum_weight: float = 2.0
    rare_cluster_temperature: float = 1.0
    maximum_root_angular_priority_rad_s: float = 8.0
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.recovery_replay_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.failure_weight,
            self.high_momentum_weight,
            self.rare_cluster_temperature,
            self.maximum_root_angular_priority_rad_s,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or not 1.0 <= self.failure_weight <= 10.0
            or not 1.0 <= self.high_momentum_weight <= 10.0
            or not 0.0 <= self.rare_cluster_temperature <= 2.0
            or not 2.0 <= self.maximum_root_angular_priority_rad_s <= 20.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery replay config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


class RecoveryReplaySampler:
    """Deterministic, cluster-balanced sampler over real physical failures."""

    def __init__(
        self,
        *,
        snapshots: Sequence[RecoverySnapshot],
        config: RecoveryReplayConfig | None = None,
    ) -> None:
        if not snapshots:
            raise ValueError("recovery replay sampler requires snapshots")
        self.snapshots = tuple(snapshots)
        self.config = config or RecoveryReplayConfig()
        bindings = {
            (
                item.body_hash,
                item.physics_scene_hash,
                item.source_policy_hash,
                item.source_config_hash,
            )
            for item in snapshots
        }
        if len(bindings) != 1:
            raise ValueError("recovery replay sampler cannot mix source contracts")
        counts = Counter(item.posture_cluster for item in self.snapshots)
        weights = []
        for item in self.snapshots:
            weight = counts[item.posture_cluster] ** (-self.config.rare_cluster_temperature)
            if item.failed:
                weight *= self.config.failure_weight
            if item.posture_cluster == "AIRBORNE_OR_HIGH_MOMENTUM":
                weight *= self.config.high_momentum_weight
            angular_speed = float(np.linalg.norm(item.qvel[3:6]))
            weight *= 1.0 + min(
                angular_speed / self.config.maximum_root_angular_priority_rad_s,
                1.0,
            )
            weights.append(weight)
        self.probability = np.asarray(weights, dtype=np.float64)
        self.probability /= self.probability.sum()
        self.binding_hash = hash_json(
            {
                "snapshots": [item.snapshot_hash for item in self.snapshots],
                "config_hash": self.config.config_hash,
            }
        )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        *,
        config: RecoveryReplayConfig | None = None,
    ) -> RecoveryReplaySampler:
        return cls(
            snapshots=load_recovery_snapshot_corpus(manifest_path),
            config=config,
        )

    def sample_indices(self, *, batch_size: int, seed: int) -> np.ndarray:
        if not 1 <= batch_size <= 65_536 or seed < 0:
            raise ValueError("recovery replay sampling request is invalid")
        generator = np.random.default_rng(seed)
        return generator.choice(
            len(self.snapshots),
            size=batch_size,
            replace=batch_size > len(self.snapshots),
            p=self.probability,
        )

    def sample(self, *, batch_size: int, seed: int) -> tuple[RecoverySnapshot, ...]:
        return tuple(self.snapshots[int(index)] for index in self.sample_indices(
            batch_size=batch_size,
            seed=seed,
        ))

    def summary(self) -> dict[str, Any]:
        clusters = Counter(item.posture_cluster for item in self.snapshots)
        stages = Counter(item.stage for item in self.snapshots)
        return {
            "schema_version": "rosclaw.recovery_replay_summary.v1",
            "snapshot_count": len(self.snapshots),
            "cluster_counts": dict(sorted(clusters.items())),
            "stage_counts": dict(sorted(stages.items())),
            "failed_fraction": sum(item.failed for item in self.snapshots) / len(self.snapshots),
            "sampling_probability_minimum": float(self.probability.min()),
            "sampling_probability_maximum": float(self.probability.max()),
            "binding_hash": self.binding_hash,
            "config_hash": self.config.config_hash,
            "activation_ceiling": "SIM_ONLY",
            "hardware_authorized": False,
        }


def _inventory(path: Path, pattern: str) -> tuple[int, str]:
    files = sorted(item for item in path.rglob(pattern) if item.is_file()) if path.is_dir() else []
    inventory = [
        {
            "path": item.relative_to(path).as_posix(),
            "bytes": item.stat().st_size,
        }
        for item in files
    ]
    return len(files), str(hash_json(inventory))


def audit_recovery_learning_sources(
    *,
    snapshot_manifest_path: Path,
    settled_snapshot_manifest_path: Path | None = None,
    motiondecode_root: Path,
    mosaic_root: Path,
    retargeted_motion_root: Path,
    opentrack_root: Path,
    protomotions_root: Path,
    robonaldo_root: Path,
) -> dict[str, Any]:
    """Map local data/frameworks to their honest role in the learning chain."""

    snapshots = load_recovery_snapshot_corpus(snapshot_manifest_path)
    settled_snapshots = (
        snapshots
        if settled_snapshot_manifest_path is None
        else load_recovery_snapshot_corpus(settled_snapshot_manifest_path)
    )
    cluster_counts = Counter(item.posture_cluster for item in snapshots)
    settled_cluster_counts = Counter(item.posture_cluster for item in settled_snapshots)
    motiondecode = motiondecode_root.expanduser().resolve()
    recovery_directories = {
        "sit_to_stand": "1.1.2.2.Sit_to_Stand",
        "prone_to_stand": "1.1.2.4.Prone_to_Stand",
        "lie_to_stand": "1.1.2.6.Lie_Down_to_Stand",
        "fall_and_recovery": "1.7.3.9.Fall_and_Fall_Recovery",
    }
    motiondecode_families: dict[str, Any] = {}
    for family, directory_name in recovery_directories.items():
        matches = list((motiondecode / "samples").rglob(directory_name))
        directory = matches[0] if matches else motiondecode / "__missing__"
        count, inventory_hash = _inventory(directory, "*.csv")
        motiondecode_families[family] = {
            "file_count": count,
            "inventory_hash": inventory_hash,
        }
    mosaic_count, mosaic_hash = _inventory(
        mosaic_root.expanduser().resolve(), "g1_stand_up_rosbag2_*.npz"
    )
    retargeted_count, retargeted_hash = _inventory(
        retargeted_motion_root.expanduser().resolve(), "fallAndGetUp*.pkl"
    )
    opentrack = opentrack_root.expanduser().resolve()
    protomotions = protomotions_root.expanduser().resolve()
    robonaldo = robonaldo_root.expanduser().resolve()
    opentrack_dagger = opentrack / "track_mj/learning/policy/dagger/dagger_horizon.py"
    protomotions_reset = protomotions / "protomotions/envs/base_env/env.py"
    robonaldo_expert = (
        robonaldo / "policy/beyondmimic_mj/model/beyondmimic_mj.onnx"
    )
    low_momentum_fallen = sum(
        count
        for cluster, count in settled_cluster_counts.items()
        if cluster
        not in {"STANDING", "AIRBORNE_OR_HIGH_MOMENTUM", "KNEELING_OR_SUPPORTED"}
    )
    posture_coverage = {
        cluster: settled_cluster_counts.get(cluster, 0)
        for cluster in ("LEFT_SIDE", "RIGHT_SIDE", "PRONE", "SUPINE")
    }
    stages = [
        {
            "stage": "A_IMPACT_ABSORPTION",
            "ready": cluster_counts.get("AIRBORNE_OR_HIGH_MOMENTUM", 0) >= 8,
            "reset_source": "ROSCLAW_TRUE_POST_SKILL_SNAPSHOTS",
            "teacher": None,
            "algorithm": "BOUNDED_RESIDUAL_ACTOR_CRITIC_WITH_FAILURE_REPLAY",
            "objective": "DISSIPATE_MOMENTUM_WITHOUT_HEAD_OR_JOINT_LIMIT_IMPACT",
        },
        {
            "stage": "B_MULTI_POSTURE_GETUP",
            "ready": low_momentum_fallen >= 8 and all(posture_coverage.values()),
            "reset_source": "SETTLED_TRUE_SNAPSHOTS_PLUS_PHYSICS_PERTURBATIONS",
            "teacher": "MOTIONDECODE_MOSAIC_LAFAN_KINEMATIC_PRIORS",
            "algorithm": "TRACKING_RL_PER_CLUSTER",
            "objective": "REACH_BILATERAL_LOW_MOMENTUM_STANDING_ENVELOPE",
        },
        {
            "stage": "C_EXPERT_DISTILLATION",
            "ready": opentrack_dagger.is_file() and robonaldo_expert.is_file(),
            "reset_source": "STAGES_A_AND_B",
            "teacher": "ROUTED_RECOVERY_EXPERTS",
            "algorithm": "OPENTRACK_STYLE_MULTI_EXPERT_DAGGER",
            "objective": "ONE_PROPRIOCEPTIVE_RECOVERY_POLICY_WITHOUT_PRIVILEGED_ROUTE",
        },
        {
            "stage": "D_FULL_CHAIN_PROMOTION",
            "ready": False,
            "reset_source": "SHOT_SAVE_LANDING_GETUP_READY_CONTINUOUS_EPISODE",
            "teacher": None,
            "algorithm": "ROSCLAW_PRACTICE_MEMORY_GROWTH_PAIRED_GATE",
            "objective": "PRESERVE_SAVE_RATE_AND_PROVE_FINAL_STABILITY",
        },
    ]
    report: dict[str, Any] = {
        "schema_version": "rosclaw.recovery_learning_source_audit.v1",
        "true_snapshot_count": len(snapshots),
        "true_snapshot_cluster_counts": dict(sorted(cluster_counts.items())),
        "settled_snapshot_count": len(settled_snapshots),
        "settled_snapshot_cluster_counts": dict(sorted(settled_cluster_counts.items())),
        "settled_snapshot_posture_coverage": posture_coverage,
        "sources": {
            "motiondecode": {
                "families": motiondecode_families,
                "role": "KINEMATIC_STYLE_AND_MULTI_POSTURE_REFERENCE_ONLY",
            },
            "mosaic": {
                "g1_stand_up_file_count": mosaic_count,
                "inventory_hash": mosaic_hash,
                "role": "NATIVE_G1_RECOVERY_MOTION_DIVERSITY",
            },
            "g1_retargeted_lafan": {
                "fall_getup_file_count": retargeted_count,
                "inventory_hash": retargeted_hash,
                "role": "SIX_RECOVERY_EXPERT_SEEDS",
            },
            "opentrack": {
                "dagger_available": opentrack_dagger.is_file(),
                "role": "MULTI_EXPERT_DAGGER_DISTILLATION",
            },
            "protomotions": {
                "recovery_reset_available": protomotions_reset.is_file(),
                "role": "RECOVERY_RESET_PATTERN_ADAPTED_TO_TRUE_SNAPSHOTS",
            },
            "robonaldo_mjlab": {
                "neural_getup_expert_available": robonaldo_expert.is_file(),
                "role": "PHYSICS_QUALIFIED_LOCAL_GETUP_EXPERT",
            },
        },
        "curriculum_stages": stages,
        "critical_gap": (
            "TRUE_SETTLED_LEFT_RIGHT_PRONE_SUPINE_POST_SAVE_COVERAGE"
            if not stages[1]["ready"]
            else None
        ),
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    return report


def write_recovery_learning_source_audit(*, output_path: Path, **inputs: Any) -> dict[str, Any]:
    """Write an audit atomically so its readiness decision is durable evidence."""

    report = audit_recovery_learning_sources(**inputs)
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return report


def build_recovery_teacher_assignments(
    *,
    snapshot_manifest_path: Path,
    motion_library_path: Path,
    motiondecode_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Route each physical reset to a posture-matched kinematic teacher."""

    snapshots = load_recovery_snapshot_corpus(snapshot_manifest_path)
    library = load_goalkeeper_motion_library(
        motion_library_path,
        dataset_root=motiondecode_root,
    )
    recovery_clips = tuple(
        clip for clip in library.clips if clip.family is GoalkeeperMotionFamily.RECOVERY
    )
    by_posture = {clip.recovery_posture: clip for clip in recovery_clips}
    assignments = []
    missing: Counter[str] = Counter()
    for snapshot in snapshots:
        teacher = by_posture.get(snapshot.posture_cluster)
        if teacher is None:
            missing[snapshot.posture_cluster] += 1
            continue
        row = {
            "snapshot_hash": snapshot.snapshot_hash,
            "snapshot_posture": snapshot.posture_cluster,
            "teacher_clip_id": teacher.clip_id,
            "teacher_source_path": teacher.source_relative_path,
            "teacher_source_hash": teacher.source_hash,
            "teacher_start_frame": teacher.segment_start_frame,
            "teacher_end_frame": teacher.segment_end_frame,
            "teacher_frame_rate_hz": teacher.source_fps,
            "authority": "TRAIN_ONLY_KINEMATIC_REFERENCE",
        }
        row["assignment_hash"] = hash_json(row)
        assignments.append(row)
    report: dict[str, Any] = {
        "schema_version": "rosclaw.recovery_teacher_assignments.v1",
        "snapshot_count": len(snapshots),
        "assigned_count": len(assignments),
        "exact_posture_assignment_rate": len(assignments) / len(snapshots),
        "missing_posture_counts": dict(sorted(missing.items())),
        "motion_library_hash": library.library_hash,
        "assignments": assignments,
        "teacher_claim_boundary": (
            "KINEMATIC_REFERENCE_NOT_PHYSICS_QUALIFIED_CONTROLLER"
        ),
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return report


__all__ = [
    "RecoveryReplayConfig",
    "RecoveryReplaySampler",
    "audit_recovery_learning_sources",
    "build_recovery_teacher_assignments",
    "write_recovery_learning_source_audit",
]
