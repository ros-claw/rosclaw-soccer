from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.impact_recovery_champion import (
    ImpactRecoveryChampionChallengeConfig,
    build_impact_recovery_champion_challenge,
)
from rosclaw_soccer.training.impact_recovery_distillation import (
    ImpactRecoveryDistillationConfig,
)
from rosclaw_soccer.training.impact_recovery_mjx import (
    ImpactRecoveryMJXConfig,
    ImpactRecoveryMJXEvaluationConfig,
)
from rosclaw_soccer.training.impact_recovery_selection import (
    ImpactRecoveryCandidate,
    ImpactRecoverySelectionConfig,
    build_impact_recovery_selection,
    validate_impact_recovery_memory_diagnostic,
    validate_impact_recovery_selection_report,
)

_DIGEST = "sha256:" + "1" * 64


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _diagnostic(path: Path, population: str, successes: int) -> Path:
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_memory_baseline_diagnostic.v1",
        "curriculum_manifest_hash": _DIGEST,
        "mode": f"{population}_BASELINE",
        "population": population,
        "num_envs": 8,
        "seeds": [1, 2],
        "episode_count": 16,
        "success_count": successes,
        "elapsed_bins": {"all": {"attempts": 16, "successes": successes}},
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(path, report)
    return path


def _training(path: Path) -> Path:
    checkpoint = path.parent / "checkpoints" / "65536" / "params"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"safe-checkpoint:{path.parent.name}".encode())
    files = [
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
        "action_semantics": "DIRECT_BOUNDED_29_JOINT_PD_RESIDUAL_AROUND_FROZEN_MEMORY",
        "checkpoint_tree_hash": hash_json(files),
        "checkpoint_files": files,
        "sealed_full_chain_holdouts_loaded": 0,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(path, report)
    return path


def _evaluation(path: Path, training_path: Path, acquisition: int, retention: int) -> Path:
    training = json.loads(training_path.read_text(encoding="utf-8"))
    config = ImpactRecoveryMJXEvaluationConfig(num_envs=8, seeds=(1, 2))
    selected_checkpoint_files = [
        {**row, "path": str(row["path"]).removeprefix("65536/")}
        for row in training["checkpoint_files"]
        if str(row["path"]).startswith("65536/")
    ]
    populations = {
        name: {
            "episode_count": 16,
            "success_count": success,
            "success_rate": success / 16,
            "repeats": [
                {
                    "seed": seed,
                    "success_count": repeat_success,
                    "success_rate": repeat_success / config.num_envs,
                }
                for seed, repeat_success in zip(
                    config.seeds,
                    (min(success, config.num_envs), max(0, success - config.num_envs)),
                    strict=True,
                )
            ],
        }
        for name, success in (("acquisition", acquisition), ("retention", retention))
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_mjx_evaluation_report.v1",
        "config": config.__dict__,
        "config_hash": config.config_hash,
        "training_report_hash": training["report_hash"],
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
    _write_json(path, report)
    return path


def _distilled_training(path: Path) -> Path:
    path.parent.mkdir(parents=True)
    corpus = path.parent / "gated-distillation-corpus.npz"
    model = path.parent / "student-model.npz"
    np.savez_compressed(corpus, value=np.zeros(1, dtype=np.float32))
    np.savez_compressed(
        model,
        input_mean=np.zeros(2, dtype=np.float32),
        input_std=np.ones(2, dtype=np.float32),
        weight=np.zeros((2, 29), dtype=np.float32),
        bias=np.zeros(29, dtype=np.float32),
    )
    config = ImpactRecoveryDistillationConfig(
        student_model_type="RIDGE_CURRENT_FRAME",
        training_steps=100,
        holdout_state_count=1,
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_distilled_student.v1",
        "config": config.__dict__,
        "config_hash": config.config_hash,
        "teacher_report_hash": _DIGEST,
        "teacher_report_file_hash": _DIGEST,
        "teacher_corpus_hash": _DIGEST,
        "curriculum_manifest_hash": _DIGEST,
        "curriculum_archive_hash": _DIGEST,
        "body_hash": _DIGEST,
        "corpus": corpus.name,
        "corpus_hash": hash_bytes(corpus.read_bytes()),
        "student_model": model.name,
        "student_model_hash": hash_bytes(model.read_bytes()),
        "training_metrics": {"validation_loss_improvement_fraction": 0.5},
        "student_exam_eligible": True,
        "residual_authority_steps": 80,
        "device_count": 4,
        "all_devices_used": True,
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(path, report)
    return path


def _distilled_evaluation(path: Path, student_path: Path, acquisition: int, retention: int) -> Path:
    student = json.loads(student_path.read_text(encoding="utf-8"))
    config = ImpactRecoveryMJXEvaluationConfig(num_envs=8, seeds=(1, 2))
    populations: dict[str, Any] = {}
    for name, success in (("acquisition", acquisition), ("retention", retention)):
        first = success // 2
        second = success - first
        populations[name] = {
            "episode_count": 16,
            "success_count": success,
            "success_rate": success / 16,
            "repeats": [
                {"seed": 1, "success_count": first, "success_rate": first / 8},
                {"seed": 2, "success_count": second, "success_rate": second / 8},
            ],
        }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_distilled_evaluation.v1",
        "config": config.__dict__,
        "config_hash": config.config_hash,
        "student_report_hash": student["report_hash"],
        "student_model_hash": student["student_model_hash"],
        "curriculum_manifest_hash": _DIGEST,
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


def test_selection_rejects_forgetting_and_selects_stable_acquisition(tmp_path: Path) -> None:
    acquisition = _diagnostic(tmp_path / "acquisition.json", "ACQUISITION", 4)
    retention = _diagnostic(tmp_path / "retention.json", "RETENTION", 14)
    weak_train = _training(tmp_path / "weak" / "training-report.json")
    strong_train = _training(tmp_path / "strong" / "training-report.json")
    weak_eval = _evaluation(tmp_path / "weak-eval.json", weak_train, 12, 8)
    strong_eval = _evaluation(tmp_path / "strong-eval.json", strong_train, 12, 13)

    report = build_impact_recovery_selection(
        acquisition_baseline_path=acquisition,
        retention_baseline_path=retention,
        candidates=(
            ImpactRecoveryCandidate("weak-retention", weak_train, weak_eval),
            ImpactRecoveryCandidate("stable-acquisition", strong_train, strong_eval),
        ),
        output_dir=tmp_path / "selection",
        source_checkout_path=tmp_path / "checkout",
        config=ImpactRecoverySelectionConfig(minimum_population_episode_count=16),
    )

    decisions = {row["candidate_id"]: row["decision"] for row in report["candidates"]}
    assert decisions == {
        "stable-acquisition": "QUALIFIED_FOR_CPU_FULL_CHAIN_EXAM",
        "weak-retention": "REJECTED",
    }
    assert report["qualified_candidate_ids"] == ["stable-acquisition"]
    assert report["promotion_eligible"] is False


def test_diagnostic_and_selection_fail_closed_on_tamper(tmp_path: Path) -> None:
    diagnostic = _diagnostic(tmp_path / "diagnostic.json", "RETENTION", 14)
    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    payload["success_count"] = 15
    _write_json(diagnostic, payload)

    with pytest.raises(ValueError, match="integrity"):
        validate_impact_recovery_memory_diagnostic(diagnostic)

    selection = tmp_path / "selection.json"
    _write_json(selection, {"schema_version": "tampered", "report_hash": _DIGEST})
    with pytest.raises(ValueError, match="incomplete"):
        validate_impact_recovery_selection_report(selection)


def test_selection_accepts_distilled_candidate_schema_but_not_extra_authority(
    tmp_path: Path,
) -> None:
    acquisition = _diagnostic(tmp_path / "acquisition.json", "ACQUISITION", 4)
    retention = _diagnostic(tmp_path / "retention.json", "RETENTION", 14)
    student = _distilled_training(tmp_path / "student" / "distillation-report.json")
    evaluation = _distilled_evaluation(tmp_path / "student-eval.json", student, 12, 13)

    report = build_impact_recovery_selection(
        acquisition_baseline_path=acquisition,
        retention_baseline_path=retention,
        candidates=(
            ImpactRecoveryCandidate(
                "distilled",
                student,
                evaluation,
                candidate_kind="DISTILLED_STUDENT",
            ),
        ),
        output_dir=tmp_path / "selection",
        source_checkout_path=tmp_path / "checkout",
        config=ImpactRecoverySelectionConfig(minimum_population_episode_count=16),
    )

    assert report["candidates"][0]["candidate_kind"] == "DISTILLED_STUDENT"
    assert report["candidates"][0]["decision"] == "QUALIFIED_FOR_CPU_FULL_CHAIN_EXAM"
    assert report["promotion_eligible"] is False


def test_champion_challenge_archives_tie_and_accepts_bounded_growth(tmp_path: Path) -> None:
    incumbent_training = _training(tmp_path / "incumbent" / "training-report.json")
    challenger_training = _training(tmp_path / "challenger" / "training-report.json")
    incumbent = _evaluation(
        tmp_path / "incumbent-eval.json", incumbent_training, acquisition=12, retention=14
    )
    tied = _evaluation(
        tmp_path / "tied-eval.json", challenger_training, acquisition=12, retention=14
    )
    improved = _evaluation(
        tmp_path / "improved-eval.json", challenger_training, acquisition=13, retention=13
    )
    config = ImpactRecoveryChampionChallengeConfig(minimum_population_episode_count=16)

    tied_report = build_impact_recovery_champion_challenge(
        incumbent_training_report_path=incumbent_training,
        incumbent_evaluation_report_path=incumbent,
        challenger_training_report_path=challenger_training,
        challenger_evaluation_report_path=tied,
        output_dir=tmp_path / "tied-challenge",
        source_checkout_path=tmp_path / "checkout",
        config=config,
    )
    improved_report = build_impact_recovery_champion_challenge(
        incumbent_training_report_path=incumbent_training,
        incumbent_evaluation_report_path=incumbent,
        challenger_training_report_path=challenger_training,
        challenger_evaluation_report_path=improved,
        output_dir=tmp_path / "improved-challenge",
        source_checkout_path=tmp_path / "checkout",
        config=config,
    )

    assert tied_report["decision"] == "CHALLENGER_ARCHIVED"
    assert improved_report["decision"] == "CHALLENGER_READY_FOR_CPU_FULL_CHAIN_EXAM"
    assert improved_report["promotion_eligible"] is False


def test_champion_challenge_rejects_checkpoint_not_bound_to_training(tmp_path: Path) -> None:
    incumbent_training = _training(tmp_path / "incumbent" / "training-report.json")
    challenger_training = _training(tmp_path / "challenger" / "training-report.json")
    incumbent = _evaluation(
        tmp_path / "incumbent-eval.json", incumbent_training, acquisition=12, retention=14
    )
    challenger = _evaluation(
        tmp_path / "challenger-eval.json", challenger_training, acquisition=13, retention=14
    )
    payload = json.loads(challenger.read_text(encoding="utf-8"))
    payload["selected_checkpoint_files"][0]["hash"] = hash_bytes(b"unbound")
    payload["selected_checkpoint_hash"] = hash_json(payload["selected_checkpoint_files"])
    payload["report_hash"] = hash_json(
        {key: value for key, value in payload.items() if key != "report_hash"}
    )
    _write_json(challenger, payload)

    with pytest.raises(ValueError, match="suite changed"):
        build_impact_recovery_champion_challenge(
            incumbent_training_report_path=incumbent_training,
            incumbent_evaluation_report_path=incumbent,
            challenger_training_report_path=challenger_training,
            challenger_evaluation_report_path=challenger,
            output_dir=tmp_path / "challenge",
            source_checkout_path=tmp_path / "checkout",
            config=ImpactRecoveryChampionChallengeConfig(minimum_population_episode_count=16),
        )
