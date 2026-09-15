from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.impact_recovery_curriculum import (
    ImpactRecoveryCurriculumConfig,
    ImpactRecoverySource,
    build_impact_recovery_curriculum,
    validate_impact_recovery_curriculum,
)
from rosclaw_soccer.training.impact_recovery_frontier import (
    ImpactRecoveryFrontierConfig,
    build_impact_recovery_frontier,
    validate_impact_recovery_frontier,
)
from rosclaw_soccer.training.impact_recovery_mjx import (
    ImpactRecoveryMJXConfig,
    ImpactRecoveryMJXEvaluationConfig,
    _memory_blend_fraction,
    _teacher_novelty_gate,
    validate_impact_recovery_mjx_evaluation_report,
    validate_impact_recovery_mjx_report,
)

_DIGEST = "sha256:" + "1" * 64


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _trajectory(*, succeeded: bool) -> dict[str, np.ndarray[Any, Any]]:
    rows = 40
    time = np.arange(rows, dtype=np.float64) * 0.02
    pose = np.zeros((rows, 7), dtype=np.float64)
    pose[:, 0] = 8.0
    pose[:, 2] = 0.79
    if succeeded:
        pose[:, 3] = 0.0
        pose[:, 6] = 1.0
    else:
        pose[:, 3] = 1.0
    root_velocity = np.zeros((rows, 6), dtype=np.float64)
    root_velocity[:, 0] = 0.05 if succeeded else -0.35
    root_velocity[:, 5] = 0.02 if succeeded else 0.8
    joint_position = np.zeros((rows, 29), dtype=np.float64)
    joint_position[:, :12] = np.linspace(0.0, 0.08 if succeeded else -0.12, rows)[:, None]
    policy_action = joint_position + (0.01 if succeeded else -0.03)
    contact = np.zeros(rows, dtype=np.bool_)
    contact[5:7] = True
    ball_pose = np.zeros((rows, 7), dtype=np.float64)
    ball_pose[:, 3] = 1.0
    return {
        "time": time,
        "second_ball_pose": ball_pose,
        "second_ball_velocity": np.zeros((rows, 6), dtype=np.float64),
        "goalkeeper_pelvis_pose": pose,
        "goalkeeper_root_velocity": root_velocity,
        "goalkeeper_foot_contact": np.ones((rows, 2), dtype=np.bool_),
        "goalkeeper_joint_position": joint_position,
        "goalkeeper_joint_velocity": np.zeros((rows, 29), dtype=np.float64),
        "goalkeeper_executed_torque": np.zeros((rows, 29), dtype=np.float64),
        "goalkeeper_policy_action": policy_action,
        "goalkeeper_kp": np.full((rows, 29), 100.0, dtype=np.float64),
        "goalkeeper_kd": np.full((rows, 29), 2.0, dtype=np.float64),
        "goalkeeper_second_ball_contact": contact,
    }


def _episode(root: Path, name: str, *, succeeded: bool) -> Path:
    directory = root / name
    directory.mkdir()
    request = {
        "schema_version": "rosclaw_soccer.role_isolated_second_striker_probe_request.v1",
        "config": {"seed": 117},
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "reproducibility_closure_hash": _DIGEST,
    }
    request["config_hash"] = hash_json(request["config"])
    request_path = directory / "request.json"
    _write_json(request_path, request)

    trajectory = _trajectory(succeeded=succeeded)
    replays: list[dict[str, Any]] = []
    for replay_index in range(2):
        trajectory_path = directory / f"trajectory-{replay_index}.npz"
        np.savez_compressed(trajectory_path, **trajectory)  # type: ignore[arg-type]
        replays.append(
            {
                "trajectory_file": trajectory_path.name,
                "trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
                "trajectory_digest": trajectory_digest(trajectory),
                "result": {"complete": True},
                "evaluation": {"passed": succeeded},
                "candidate_diagnostics": {"ready": succeeded},
            }
        )
    evidence = {
        "schema_version": "rosclaw_soccer.role_isolated_second_striker_probe_evidence.v1",
        "candidate_promoted": succeeded,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
        "request_hash": hash_bytes(request_path.read_bytes()),
        "reproducibility_closure_hash": _DIGEST,
        "replays": replays,
    }
    evidence["report_hash"] = hash_json(evidence)
    evidence_path = directory / "evidence.json"
    _write_json(evidence_path, evidence)
    return evidence_path


def _build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    dynamic_gain: bool = False,
    teacher_state: bool = False,
    phase_search: bool = False,
) -> tuple[Path, dict[str, Any]]:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    success = _episode(evidence_root, "success", succeeded=True)
    failure = _episode(evidence_root, "failure", succeeded=False)
    asset_root = tmp_path / "assets"
    asset_dir = asset_root / "g1_description"
    asset_dir.mkdir(parents=True)
    (asset_dir / "g1_liao.xml").write_text("<mujoco/>", encoding="utf-8")
    (asset_dir / "scene_with_ball.xml").write_text("<mujoco/>", encoding="utf-8")
    monkeypatch.setattr(
        "rosclaw_soccer.training.impact_recovery_curriculum.g1_body_hash",
        lambda _: "sha256:" + "2" * 64,
    )
    output = tmp_path / "curriculum"
    report = build_impact_recovery_curriculum(
        sources=(
            ImpactRecoverySource("retention", success, "RETENTION_ANCHOR", True),
            ImpactRecoverySource("hard-failure", failure, "ACQUISITION_FAILURE", False),
        ),
        asset_root=asset_root,
        source_checkout=tmp_path / "checkout",
        output_dir=output,
        config=ImpactRecoveryCurriculumConfig(
            first_offset_sec=0.10,
            last_offset_sec=0.30,
            sample_stride_sec=0.10,
            control_dt_sec=0.02,
            memory_horizon_steps=8,
            dynamic_gain_memory_enabled=dynamic_gain,
            teacher_state_memory_enabled=teacher_state,
            teacher_phase_search_radius_sec=0.20 if phase_search else 0.0,
        ),
    )
    return output, report


def test_builds_content_bound_failure_curriculum_without_failed_teacher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, report = _build(tmp_path, monkeypatch)

    assert report["snapshot_count"] == 6
    assert report["retention_snapshot_count"] == 3
    assert report["acquisition_failure_snapshot_count"] == 3
    assert report["teacher_memory_sources"] == ["retention"]
    assert report["failed_sources_used_as_teacher_count"] == 0
    assert report["historical_source_closures_recomputed"] is False
    with np.load(output / "impact-recovery-curriculum.npz", allow_pickle=False) as archive:
        assert archive["qpos"].shape == (6, 36)
        assert archive["qvel"].shape == (6, 35)
        assert archive["frozen_memory_target_rad"].shape == (6, 8, 29)
        assert archive["initial_motor_target_rad"].shape == (6, 29)
        assert np.array_equal(archive["memory_route_index"], np.zeros(6, dtype=np.int32))
        assert np.array_equal(
            archive["source_succeeded"],
            np.asarray((True, True, True, False, False, False)),
        )
        # The current target produced the reset row; continuation starts with
        # the next recorded target instead of replaying a consumed command.
        assert np.all(
            archive["frozen_memory_target_rad"][0, 0, :12]
            > archive["initial_motor_target_rad"][0, :12]
        )
    assert report["schema_version"] == "rosclaw_soccer.impact_recovery_curriculum.v3"
    assert report["teacher_memory_semantics"] == "RECORDED_SUCCESSFUL_NEXT_PD_MOTOR_TARGET_SEQUENCE"


def test_validator_rejects_archive_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _ = _build(tmp_path, monkeypatch)
    archive = output / "impact-recovery-curriculum.npz"
    archive.write_bytes(archive.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="integrity"):
        validate_impact_recovery_curriculum(output / "impact-recovery-curriculum.json")


def test_dynamic_gain_curriculum_binds_teacher_and_reset_gains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, report = _build(tmp_path, monkeypatch, dynamic_gain=True)

    assert report["schema_version"] == "rosclaw_soccer.impact_recovery_curriculum.v4"
    assert report["teacher_gain_semantics"] == "RECORDED_SUCCESSFUL_NEXT_DYNAMIC_PD_GAINS"
    with np.load(output / "impact-recovery-curriculum.npz", allow_pickle=False) as archive:
        assert archive["initial_kp"].shape == (6, 29)
        assert archive["initial_kd"].shape == (6, 29)
        assert archive["frozen_memory_kp"].shape == (6, 8, 29)
        assert archive["frozen_memory_kd"].shape == (6, 8, 29)
        assert np.all(archive["frozen_memory_kp"] > 0.0)


def test_teacher_state_curriculum_binds_causal_proprioception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, report = _build(tmp_path, monkeypatch, dynamic_gain=True, teacher_state=True)

    assert report["schema_version"] == "rosclaw_soccer.impact_recovery_curriculum.v5"
    assert report["teacher_state_semantics"] == "RECORDED_SUCCESSFUL_CAUSAL_PROPRIOCEPTIVE_SEQUENCE"
    with np.load(output / "impact-recovery-curriculum.npz", allow_pickle=False) as archive:
        assert archive["frozen_memory_qpos"].shape == (6, 8, 36)
        assert archive["frozen_memory_qvel"].shape == (6, 8, 35)


def test_phase_aligned_curriculum_binds_nearest_successful_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, report = _build(
        tmp_path,
        monkeypatch,
        dynamic_gain=True,
        teacher_state=True,
        phase_search=True,
    )

    assert report["schema_version"] == "rosclaw_soccer.impact_recovery_curriculum.v6"
    assert (
        report["teacher_retrieval_semantics"]
        == "NEAREST_SUCCESSFUL_STATE_WITH_BOUNDED_PHASE_SEARCH"
    )
    assert all(isinstance(row["memory_route_reference_frame"], int) for row in report["rows"])
    assert all(abs(row["memory_route_phase_offset_sec"]) <= 0.22 for row in report["rows"])


def test_source_outcome_declaration_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    success = _episode(evidence_root, "success", succeeded=True)
    failure = _episode(evidence_root, "failure", succeeded=False)
    asset_root = tmp_path / "assets"
    asset_dir = asset_root / "g1_description"
    asset_dir.mkdir(parents=True)
    (asset_dir / "g1_liao.xml").write_text("<mujoco/>", encoding="utf-8")
    (asset_dir / "scene_with_ball.xml").write_text("<mujoco/>", encoding="utf-8")
    monkeypatch.setattr(
        "rosclaw_soccer.training.impact_recovery_curriculum.g1_body_hash",
        lambda _: "sha256:" + "2" * 64,
    )

    with pytest.raises(ValueError, match="outcome"):
        build_impact_recovery_curriculum(
            sources=(
                ImpactRecoverySource("claimed-success", failure, "RETENTION_ANCHOR", True),
                ImpactRecoverySource("failure", success, "ACQUISITION_FAILURE", False),
            ),
            asset_root=asset_root,
            source_checkout=tmp_path / "checkout",
            output_dir=tmp_path / "curriculum",
        )


def test_config_rejects_hardware_authority() -> None:
    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryCurriculumConfig(hardware_authorized=True)

    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryMJXConfig(learning_stage="UNKNOWN")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryMJXConfig(retention_memory_mode="UNKNOWN")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryMJXConfig(gain_memory_mode="UNKNOWN")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryCurriculumConfig(teacher_state_memory_enabled=True)

    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryCurriculumConfig(teacher_phase_search_radius_sec=0.5)

    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryMJXConfig(residual_gate_mode="TEACHER_NOVELTY")

    with pytest.raises(ValueError, match="invalid"):
        ImpactRecoveryMJXConfig(residual_authority_steps=5)


def test_direct_retention_memory_does_not_delay_successful_teacher() -> None:
    direct = ImpactRecoveryMJXConfig(retention_memory_mode="DIRECT_REPLAY")
    blended = ImpactRecoveryMJXConfig(retention_memory_mode="BLENDED")
    step = jnp.asarray(1, dtype=jnp.int32)

    assert float(_memory_blend_fraction(step, jnp.asarray(False), direct)) == 1.0
    assert float(_memory_blend_fraction(step, jnp.asarray(True), direct)) == pytest.approx(1 / 75)
    assert float(_memory_blend_fraction(step, jnp.asarray(False), blended)) == pytest.approx(1 / 75)


def test_teacher_novelty_gate_freezes_memory_and_opens_off_manifold() -> None:
    config = ImpactRecoveryMJXConfig(
        gain_memory_mode="DYNAMIC",
        residual_gate_mode="TEACHER_NOVELTY",
    )
    reference_qpos = jnp.zeros(36, dtype=jnp.float32).at[2].set(0.75).at[3].set(1.0)
    reference_qvel = jnp.zeros(35, dtype=jnp.float32)

    on_manifold = _teacher_novelty_gate(
        reference_qpos,
        reference_qvel,
        reference_qpos,
        reference_qvel,
        config,
    )
    far_from_teacher = _teacher_novelty_gate(
        reference_qpos.at[2].set(0.85),
        reference_qvel,
        reference_qpos,
        reference_qvel,
        config,
    )

    assert float(on_manifold) == 0.0
    assert float(far_from_teacher) == 1.0


def _mjx_report(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoints" / "65536" / "params"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"safe-checkpoint")
    rows = [
        {
            "path": "65536/params",
            "size_bytes": checkpoint.stat().st_size,
            "hash": hash_bytes(checkpoint.read_bytes()),
        }
    ]
    config = ImpactRecoveryMJXConfig(total_timesteps=65_536, num_evals=2)
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_mjx_training_report.v2",
        "config": config.__dict__,
        "config_hash": config.config_hash,
        "curriculum_manifest_hash": _DIGEST,
        "curriculum_archive_hash": _DIGEST,
        "body_hash": _DIGEST,
        "compiled_model_contract": {"model_hash": _DIGEST},
        "rollout_backend": "MUJOCO_MJX",
        "physics_truth_backend": "CPU_MUJOCO",
        "parallelization": "BRAX_PPO_JAX_PMAP_VMAP",
        "devices": ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        "training_reset_population": "MIXED_FAILURE_PRIORITIZED",
        "evaluation_reset_population": "ACQUISITION_FAILURE_ONLY",
        "learning_stage": "BALANCE",
        "continued_from_checkpoint": False,
        "parent_checkpoint_hash": None,
        "failed_sources_used_as_teacher_count": 0,
        "actor_observation_dim": config.observation_dim,
        "actor_observation": "DEPLOYABLE_PROPRIOCEPTION_HISTORY_AND_GOAL_HEADING",
        "action_semantics": ("DIRECT_BOUNDED_29_JOINT_PD_RESIDUAL_AROUND_FROZEN_MEMORY"),
        "checkpoint_tree_hash": hash_json(rows),
        "checkpoint_files": rows,
        "sealed_full_chain_holdouts_loaded": 0,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    report_path = tmp_path / "training-report.json"
    _write_json(report_path, report)
    return report_path


def test_mjx_report_is_content_bound_and_sim_only(tmp_path: Path) -> None:
    report_path = _mjx_report(tmp_path)

    report = validate_impact_recovery_mjx_report(report_path)

    assert report["report_hash"].startswith("sha256:")
    assert report["promotion_authority"] == "NONE"
    assert report["hardware_command_sent"] is False


def test_mjx_report_rejects_checkpoint_tampering(tmp_path: Path) -> None:
    report_path = _mjx_report(tmp_path)
    (tmp_path / "checkpoints" / "65536" / "params").write_bytes(b"changed")

    with pytest.raises(ValueError, match="integrity"):
        validate_impact_recovery_mjx_report(report_path)


def test_expanded_evaluation_report_is_sim_only(tmp_path: Path) -> None:
    config = ImpactRecoveryMJXEvaluationConfig(num_envs=8, seeds=(1, 2))
    selected_checkpoint_files = [{"path": "params", "size_bytes": 4, "hash": hash_bytes(b"safe")}]
    populations = {
        name: {
            "episode_count": 16,
            "success_count": success_count,
            "success_rate": success_count / 16,
            "repeats": [
                {
                    "seed": seed,
                    "success_count": repeat_success,
                    "success_rate": repeat_success / config.num_envs,
                }
                for seed, repeat_success in zip(
                    config.seeds,
                    (
                        min(success_count, config.num_envs),
                        max(0, success_count - config.num_envs),
                    ),
                    strict=True,
                )
            ],
        }
        for name, success_count in (("acquisition", 7), ("retention", 15))
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_mjx_evaluation_report.v1",
        "config": config.__dict__,
        "config_hash": config.config_hash,
        "training_report_hash": _DIGEST,
        "curriculum_manifest_hash": _DIGEST,
        "selected_checkpoint_hash": hash_json(selected_checkpoint_files),
        "selected_checkpoint_files": selected_checkpoint_files,
        "populations": populations,
        "physics_backend": "MUJOCO_MJX",
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    path = tmp_path / "evaluation-report.json"
    _write_json(path, report)

    validated = validate_impact_recovery_mjx_evaluation_report(path)

    assert validated["populations"]["acquisition"]["success_rate"] == 7 / 16
    assert validated["promotion_authority"] == "NONE"


def test_expanded_evaluation_rejects_manifest_and_repeat_tampering(tmp_path: Path) -> None:
    config = ImpactRecoveryMJXEvaluationConfig(num_envs=8, seeds=(1, 2))
    checkpoint_files = [{"path": "params", "size_bytes": 4, "hash": hash_bytes(b"safe")}]
    populations = {
        name: {
            "episode_count": 16,
            "success_count": 8,
            "success_rate": 0.5,
            "repeats": [
                {"seed": seed, "success_count": 4, "success_rate": 0.5} for seed in config.seeds
            ],
        }
        for name in ("acquisition", "retention")
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_mjx_evaluation_report.v1",
        "config": config.__dict__,
        "config_hash": config.config_hash,
        "training_report_hash": _DIGEST,
        "curriculum_manifest_hash": _DIGEST,
        "selected_checkpoint_hash": hash_json(checkpoint_files),
        "selected_checkpoint_files": checkpoint_files,
        "populations": populations,
        "physics_backend": "MUJOCO_MJX",
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    path = tmp_path / "evaluation-report.json"
    _write_json(path, report)

    report["selected_checkpoint_hash"] = _DIGEST
    report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
    _write_json(path, report)
    with pytest.raises(ValueError, match="integrity"):
        validate_impact_recovery_mjx_evaluation_report(path)

    report["selected_checkpoint_hash"] = hash_json(checkpoint_files)
    report["populations"]["acquisition"]["repeats"][0]["success_count"] = 3
    report["populations"]["acquisition"]["repeats"][0]["success_rate"] = 3 / 8
    report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
    _write_json(path, report)
    with pytest.raises(ValueError, match="integrity"):
        validate_impact_recovery_mjx_evaluation_report(path)


def _frontier_evaluation_report(
    path: Path,
    *,
    curriculum_manifest_hash: str,
    acquisition_rows: list[dict[str, Any]],
) -> Path:
    config = ImpactRecoveryMJXEvaluationConfig(num_envs=8, seeds=(1, 2))
    selected_checkpoint_files = [{"path": "params", "size_bytes": 4, "hash": hash_bytes(b"safe")}]
    row_cycle = acquisition_rows * 6
    row_cycle = row_cycle[:16]
    acquisition_repeats: list[dict[str, Any]] = []
    for repeat_index, seed in enumerate(config.seeds):
        selected = row_cycle[repeat_index * 8 : (repeat_index + 1) * 8]
        success = [
            float(index % 3 != 0) for index in range(repeat_index * 8, (repeat_index + 1) * 8)
        ]
        acquisition_repeats.append(
            {
                "seed": seed,
                "success_count": int(sum(success)),
                "success_rate": sum(success) / 8,
                "episode_metrics": {
                    "curriculum_row_once": [float(row["archive_row"] + 1) for row in selected],
                    "elapsed_since_contact_once": [
                        float(row["elapsed_since_contact_sec"]) for row in selected
                    ],
                    "success": success,
                },
                "mean_episode_length": 50.0,
            }
        )
    acquisition_success = sum(int(row["success_count"]) for row in acquisition_repeats)
    populations = {
        "acquisition": {
            "episode_count": 16,
            "success_count": acquisition_success,
            "success_rate": acquisition_success / 16,
            "repeats": acquisition_repeats,
        },
        "retention": {
            "episode_count": 16,
            "success_count": 16,
            "success_rate": 1.0,
            "repeats": [
                {"seed": seed, "success_count": 8, "success_rate": 1.0} for seed in config.seeds
            ],
        },
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_mjx_evaluation_report.v1",
        "config": config.__dict__,
        "config_hash": config.config_hash,
        "training_report_hash": _DIGEST,
        "curriculum_manifest_hash": curriculum_manifest_hash,
        "selected_checkpoint_hash": hash_json(selected_checkpoint_files),
        "selected_checkpoint_files": selected_checkpoint_files,
        "populations": populations,
        "physics_backend": "MUJOCO_MJX",
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(path, report)
    return path


def test_builds_content_bound_failure_frontier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curriculum_dir, curriculum = _build(tmp_path, monkeypatch)
    acquisition_rows = [
        row for row in curriculum["rows"] if row["source_use"] == "ACQUISITION_FAILURE"
    ]
    evaluation_path = _frontier_evaluation_report(
        tmp_path / "evaluation-report.json",
        curriculum_manifest_hash=curriculum["manifest_hash"],
        acquisition_rows=acquisition_rows,
    )

    frontier = build_impact_recovery_frontier(
        curriculum_manifest_path=curriculum_dir / "impact-recovery-curriculum.json",
        evaluation_report_path=evaluation_path,
        output_dir=tmp_path / "frontier",
        source_checkout_path=tmp_path / "checkout",
        config=ImpactRecoveryFrontierConfig(elapsed_bin_width_sec=0.25),
    )

    assert frontier["row_count"] == 3
    assert frontier["observed_episode_count"] == 16
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert frontier["source_checkpoint_hash"] == evaluation["selected_checkpoint_hash"]
    assert sum(row["sampling_probability"] for row in frontier["rows"]) == pytest.approx(1.0)
    assert frontier["training_use_only"] is True
    assert frontier["promotion_authority"] == "NONE"


def test_frontier_validator_rejects_weight_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curriculum_dir, curriculum = _build(tmp_path, monkeypatch)
    acquisition_rows = [
        row for row in curriculum["rows"] if row["source_use"] == "ACQUISITION_FAILURE"
    ]
    evaluation_path = _frontier_evaluation_report(
        tmp_path / "evaluation-report.json",
        curriculum_manifest_hash=curriculum["manifest_hash"],
        acquisition_rows=acquisition_rows,
    )
    build_impact_recovery_frontier(
        curriculum_manifest_path=curriculum_dir / "impact-recovery-curriculum.json",
        evaluation_report_path=evaluation_path,
        output_dir=tmp_path / "frontier",
        source_checkout_path=tmp_path / "checkout",
    )
    path = tmp_path / "frontier" / "impact-recovery-frontier.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["sampling_probability"] *= 2.0
    _write_json(path, payload)

    with pytest.raises(ValueError, match="integrity"):
        validate_impact_recovery_frontier(path)
