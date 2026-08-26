from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.recovery_corrective_scale import (
    RecoveryCorrectiveScaleConfig,
    validate_recovery_corrective_frozen_exam_evidence,
    validate_recovery_corrective_scale_evidence,
    write_recovery_corrective_frozen_exam_evidence,
    write_recovery_corrective_scale_evidence,
)


def _write_bank(path: Path, *, count: int, seed: int, value_offset: float) -> Path:
    path.mkdir()
    step = np.resize(np.asarray((200, 300), dtype=np.int32), count)
    row = np.arange(count, dtype=np.float32)[:, None] + np.float32(value_offset)
    arrays = {
        "qpos": np.repeat(row, 36, axis=1),
        "qvel": np.repeat(row * np.float32(0.1), 35, axis=1),
        "control_step": step,
        "environment_index": np.arange(count, dtype=np.int32) % (count // 2),
        "handoff_frozen": np.zeros((count,), dtype=np.bool_),
        "trajectory_step": np.arange(count, dtype=np.int32) + 6_400,
        "trajectory_initial_step": np.full((count,), 6_230, dtype=np.int32),
        "root_body_backward_speed_mps": np.zeros((count,), dtype=np.float32),
        "root_body_lateral_speed_mps": np.zeros((count,), dtype=np.float32),
        "pelvis_yaw_speed_rad_s": np.zeros((count,), dtype=np.float32),
        "last_motor_targets": np.zeros((count, 29), dtype=np.float32),
        "last_teacher_action": np.zeros((count, 29), dtype=np.float32),
        "last_residual": np.zeros((count, 29), dtype=np.float32),
        "proprioception_history": np.zeros((count, 4, 96), dtype=np.float32),
        "phase_repeat": np.zeros((count,), dtype=np.int32),
    }
    archive_path = path / "failure-window-states.npz"
    np.savez_compressed(archive_path, **arrays)
    digest = "sha256:" + "a" * 64
    manifest = {
        "schema_version": "rosclaw_soccer.recovery_mjx_failure_state_manifest.v2",
        "config": {"num_environments": count // 2, "random_seed": seed},
        "source_failure_window_plan_hash": digest,
        "source_failure_window_plan_file_hash": digest,
        "source_training_report_hash": digest,
        "source_actor_checkpoint_hash": digest,
        "source_actor_config_hash": digest,
        "source_route_manifest_hash": digest,
        "source_route_group_hash": digest,
        "teacher_checkpoint_hash": digest,
        "motion_archive_hash": digest,
        "snapshot_manifest_hash": digest,
        "compiled_model_contract": {},
        "rollout_backend": "MUJOCO_MJX",
        "physics_truth_backend": "CPU_MUJOCO_REQUIRED_FOR_PROMOTION",
        "deterministic_actor": True,
        "full_route_reset": True,
        "requested_collection_steps": [200, 300],
        "collected_state_count": count,
        "state_archive": archive_path.name,
        "state_archive_hash": hash_bytes(archive_path.read_bytes()),
        "qpos_shape": [count, 36],
        "qvel_shape": [count, 35],
        "proprioception_history_shape": [count, 4, 96],
        "context_features_collected": [
            "qpos",
            "qvel",
            "trajectory_step",
            "trajectory_initial_step",
            "handoff_frozen",
            "last_motor_targets",
            "last_teacher_action",
            "last_residual",
            "proprioception_history",
            "phase_repeat",
        ],
        "curriculum_use_only": True,
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    manifest["report_hash"] = hash_json(manifest)
    manifest_path = path / "failure-state-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_scale_evidence_requires_novel_balanced_sources(tmp_path: Path) -> None:
    frozen = _write_bank(tmp_path / "frozen", count=8, seed=1, value_offset=0.0)
    training = _write_bank(tmp_path / "training", count=16, seed=2, value_offset=100.0)
    output = tmp_path / "scale.json"
    config = RecoveryCorrectiveScaleConfig(
        minimum_frozen_state_count=8,
        minimum_training_state_count=12,
        required_window_count=2,
        minimum_training_states_per_window=6,
    )
    report = write_recovery_corrective_scale_evidence(
        frozen_manifest_path=frozen,
        training_manifest_path=training,
        output_path=output,
        config=config,
    )
    assert report["passed"] is True
    assert report["exact_cross_bank_overlap_count"] == 0
    assert report["training_source_novel_fraction"] == 1.0
    assert report["training_unique_state_count"] == 16

    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["promotion_eligible"] = True
    tampered.pop("report_hash")
    tampered["report_hash"] = hash_json(tampered)
    output.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_corrective_scale_evidence(output)


def test_scale_evidence_rejects_exact_replay(tmp_path: Path) -> None:
    frozen = _write_bank(tmp_path / "frozen", count=8, seed=1, value_offset=0.0)
    training = _write_bank(tmp_path / "training", count=16, seed=2, value_offset=0.0)
    with pytest.raises(ValueError, match="invalid"):
        write_recovery_corrective_scale_evidence(
            frozen_manifest_path=frozen,
            training_manifest_path=training,
            output_path=tmp_path / "scale.json",
            config=RecoveryCorrectiveScaleConfig(
                minimum_frozen_state_count=8,
                minimum_training_state_count=12,
                required_window_count=2,
                minimum_training_states_per_window=6,
            ),
        )


def test_frozen_exam_recomputes_both_retention_gates(tmp_path: Path) -> None:
    digest = "sha256:" + "b" * 64
    shared = {
        "parent_checkpoint_hash": digest,
        "teacher_checkpoint_hash": digest,
        "snapshot_manifest_hash": digest,
        "route_manifest_hash": digest,
        "route_group_hash": digest,
    }
    candidate = {
        **shared,
        "report_hash": "sha256:" + "c" * 64,
        "model_archive_hash": "sha256:" + "d" * 64,
    }
    frozen = {
        **shared,
        "report_hash": "sha256:" + "e" * 64,
        "corpus_archive_hash": "sha256:" + "f" * 64,
        "failure_state_manifest_hash": "sha256:" + "1" * 64,
    }
    candidate_path = tmp_path / "candidate.json"
    frozen_path = tmp_path / "frozen.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    failure_exam = {
        "passed": True,
        "stability_retention_passed": True,
        "route_kind": "UNSEEN_EXACT_FAILURE_STATES",
    }
    normal_exam = {
        "passed": True,
        "stability_retention_passed": True,
        "route_kind": "NORMAL_PARENT_ROUTE",
    }
    output = tmp_path / "frozen-exam.json"
    report = write_recovery_corrective_frozen_exam_evidence(
        candidate_report=candidate,
        candidate_report_path=candidate_path,
        frozen_report=frozen,
        frozen_report_path=frozen_path,
        failure_exam=failure_exam,
        normal_exam=normal_exam,
        output_path=output,
    )
    assert report["frozen_benchmark_passed"] is True

    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["normal_route_paired_physics_exam"]["passed"] = False
    tampered.pop("report_hash")
    tampered["report_hash"] = hash_json(tampered)
    output.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        validate_recovery_corrective_frozen_exam_evidence(output)
