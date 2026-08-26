"""Fail-closed evidence for expanding a corrective recovery curriculum.

Training for longer on a fixed state bank is not curriculum expansion.  This
module binds an enlarged bank to a frozen benchmark, checks exact state-source
novelty, and records balanced temporal-window coverage before the new bank may
be used for a scale experiment.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.recovery_mjx import (
    validate_recovery_mjx_failure_state_manifest,
)

_LINEAGE_FIELDS = (
    "source_actor_checkpoint_hash",
    "source_actor_config_hash",
    "source_failure_window_plan_hash",
    "source_route_manifest_hash",
    "source_route_group_hash",
    "teacher_checkpoint_hash",
    "snapshot_manifest_hash",
)
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_STUDENT_SHARED_LINEAGE = (
    "parent_checkpoint_hash",
    "teacher_checkpoint_hash",
    "snapshot_manifest_hash",
    "route_manifest_hash",
    "route_group_hash",
)


@dataclass(frozen=True)
class RecoveryCorrectiveScaleConfig:
    """Minimum independent-source coverage for one scale stage."""

    minimum_frozen_state_count: int = 96
    minimum_training_state_count: int = 192
    required_window_count: int = 6
    minimum_training_states_per_window: int = 32
    maximum_exact_overlap_fraction: float = 0.0
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_corrective_scale_config.v1"

    def __post_init__(self) -> None:
        if (
            not 8 <= self.minimum_frozen_state_count <= 10_000
            or not self.minimum_frozen_state_count < self.minimum_training_state_count <= 20_000
            or not 2 <= self.required_window_count <= 32
            or not 2 <= self.minimum_training_states_per_window <= 1_024
            or not math.isfinite(self.maximum_exact_overlap_fraction)
            or not 0.0 <= self.maximum_exact_overlap_fraction <= 0.25
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.schema_version != "rosclaw_soccer.recovery_corrective_scale_config.v1"
        ):
            raise ValueError("recovery corrective scale config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _state_rows(
    manifest_path: Path,
) -> tuple[dict[str, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    manifest = validate_recovery_mjx_failure_state_manifest(manifest_path)
    archive_path = manifest_path.parent / str(manifest["state_archive"])
    with np.load(archive_path, allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float32)
        qvel = np.asarray(archive["qvel"], dtype=np.float32)
        control_step = np.asarray(archive["control_step"], dtype=np.int32)
    if (
        qpos.ndim != 2
        or qvel.ndim != 2
        or control_step.ndim != 1
        or qpos.shape[0] != qvel.shape[0]
        or qpos.shape[0] != control_step.size
        or qpos.shape[0] != manifest["collected_state_count"]
        or not np.all(np.isfinite(qpos))
        or not np.all(np.isfinite(qvel))
    ):
        raise ValueError("recovery corrective scale state archive is invalid")
    rows = np.concatenate((qpos, qvel), axis=1)
    return manifest, rows, control_step


def _row_fingerprints(rows: np.ndarray[Any, Any], control_step: np.ndarray[Any, Any]) -> set[str]:
    return {
        hash_bytes(
            np.ascontiguousarray(row).tobytes() + np.asarray((step,), dtype=np.int32).tobytes()
        )
        for row, step in zip(rows, control_step, strict=True)
    }


def write_recovery_corrective_scale_evidence(
    *,
    frozen_manifest_path: Path,
    training_manifest_path: Path,
    output_path: Path,
    config: RecoveryCorrectiveScaleConfig | None = None,
) -> dict[str, Any]:
    """Prove that a larger state bank is source-novel and window-balanced."""

    active = config or RecoveryCorrectiveScaleConfig()
    frozen_path = frozen_manifest_path.expanduser().resolve()
    training_path = training_manifest_path.expanduser().resolve()
    destination = output_path.expanduser().resolve()
    if destination.exists() or frozen_path == training_path:
        raise ValueError("recovery corrective scale evidence paths are invalid")
    frozen, frozen_rows, frozen_steps = _state_rows(frozen_path)
    training, training_rows, training_steps = _state_rows(training_path)
    lineage_matched = all(frozen.get(name) == training.get(name) for name in _LINEAGE_FIELDS)
    frozen_fingerprints = _row_fingerprints(frozen_rows, frozen_steps)
    training_fingerprints = _row_fingerprints(training_rows, training_steps)
    exact_overlap = frozen_fingerprints & training_fingerprints
    overlap_fraction = len(exact_overlap) / max(len(training_fingerprints), 1)
    windows = sorted({int(value) for value in training_steps.tolist()})
    per_window = [
        {
            "control_step": step,
            "state_count": int(np.sum(training_steps == step)),
            "unique_state_count": int(
                len(
                    _row_fingerprints(
                        training_rows[training_steps == step],
                        training_steps[training_steps == step],
                    )
                )
            ),
        }
        for step in windows
    ]
    passed = bool(
        lineage_matched
        and len(frozen_fingerprints) >= active.minimum_frozen_state_count
        and len(training_fingerprints) >= active.minimum_training_state_count
        and len(windows) == active.required_window_count
        and all(
            row["unique_state_count"] >= active.minimum_training_states_per_window
            for row in per_window
        )
        and overlap_fraction <= active.maximum_exact_overlap_fraction
        and frozen.get("config", {}).get("random_seed")
        != training.get("config", {}).get("random_seed")
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_corrective_scale_evidence.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "passed": passed,
        "lineage_matched": lineage_matched,
        "frozen_manifest_hash": frozen["report_hash"],
        "frozen_manifest_file_hash": hash_bytes(frozen_path.read_bytes()),
        "training_manifest_hash": training["report_hash"],
        "training_manifest_file_hash": hash_bytes(training_path.read_bytes()),
        "frozen_random_seed": int(frozen["config"]["random_seed"]),
        "training_random_seed": int(training["config"]["random_seed"]),
        "frozen_state_count": int(frozen_rows.shape[0]),
        "frozen_unique_state_count": len(frozen_fingerprints),
        "training_state_count": int(training_rows.shape[0]),
        "training_unique_state_count": len(training_fingerprints),
        "exact_cross_bank_overlap_count": len(exact_overlap),
        "exact_cross_bank_overlap_fraction": overlap_fraction,
        "training_source_novel_fraction": 1.0 - overlap_fraction,
        "per_failure_window": per_window,
        "claim_boundary": "SOURCE_NOVELTY_AND_TEMPORAL_COVERAGE_ONLY",
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(destination, report)
    return validate_recovery_corrective_scale_evidence(destination)


def validate_recovery_corrective_scale_evidence(path: Path) -> dict[str, Any]:
    """Validate immutable scale evidence and refuse raised authority."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery corrective scale evidence is invalid")
    report_hash = payload.pop("report_hash", None)
    valid_hash = report_hash == hash_json(payload)
    payload["report_hash"] = report_hash
    config_payload = payload.get("config")
    try:
        config = (
            RecoveryCorrectiveScaleConfig(**config_payload)
            if isinstance(config_payload, dict)
            else None
        )
    except (TypeError, ValueError):
        config = None
    rows = payload.get("per_failure_window")
    overlap_fraction = payload.get("exact_cross_bank_overlap_fraction")
    overlap_count = payload.get("exact_cross_bank_overlap_count")
    training_unique_count = payload.get("training_unique_state_count")
    summaries_valid = bool(
        config is not None
        and payload.get("config_hash") == config.config_hash
        and isinstance(rows, list)
        and len(rows) == config.required_window_count
        and sum(int(row.get("state_count", -1)) for row in rows if isinstance(row, dict))
        == payload.get("training_state_count")
        and all(
            isinstance(row, dict)
            and row.get("unique_state_count") == row.get("state_count")
            and row.get("unique_state_count", 0) >= config.minimum_training_states_per_window
            for row in rows
        )
        and payload.get("frozen_unique_state_count", 0) >= config.minimum_frozen_state_count
        and isinstance(training_unique_count, int)
        and not isinstance(training_unique_count, bool)
        and training_unique_count >= config.minimum_training_state_count
        and isinstance(overlap_count, int)
        and not isinstance(overlap_count, bool)
        and overlap_count >= 0
        and isinstance(overlap_fraction, (int, float))
        and not isinstance(overlap_fraction, bool)
        and overlap_fraction <= config.maximum_exact_overlap_fraction
        and math.isclose(
            float(overlap_fraction),
            overlap_count / training_unique_count,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(payload.get("training_source_novel_fraction", -1.0)),
            1.0 - float(payload.get("exact_cross_bank_overlap_fraction", 2.0)),
            abs_tol=1.0e-12,
        )
        and payload.get("frozen_random_seed") != payload.get("training_random_seed")
        and payload.get("lineage_matched") is True
        and payload.get("passed") is True
    )
    authority_valid = bool(
        payload.get("schema_version") == "rosclaw_soccer.recovery_corrective_scale_evidence.v1"
        and payload.get("claim_boundary") == "SOURCE_NOVELTY_AND_TEMPORAL_COVERAGE_ONLY"
        and payload.get("deployment_candidate") is False
        and payload.get("promotion_eligible") is False
        and payload.get("promotion_authority") == "NONE"
        and payload.get("activation_ceiling") == "SIM_ONLY"
        and payload.get("hardware_authorized") is False
        and payload.get("hardware_command_sent") is False
    )
    if not (valid_hash and summaries_valid and authority_valid):
        raise ValueError("recovery corrective scale evidence is invalid")
    return payload


def write_recovery_corrective_frozen_exam_evidence(
    *,
    candidate_report: Mapping[str, Any],
    candidate_report_path: Path,
    frozen_report: Mapping[str, Any],
    frozen_report_path: Path,
    failure_exam: Mapping[str, Any],
    normal_exam: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Bind a new candidate to an older, immutable source-disjoint exam."""

    candidate_path = candidate_report_path.expanduser().resolve()
    frozen_path = frozen_report_path.expanduser().resolve()
    destination = output_path.expanduser().resolve()
    if destination.exists() or candidate_path == frozen_path:
        raise ValueError("recovery corrective frozen exam paths are invalid")
    if any(
        candidate_report.get(name) != frozen_report.get(name) for name in _STUDENT_SHARED_LINEAGE
    ):
        raise ValueError("recovery corrective frozen exam lineages differ")
    candidate_hash = candidate_report.get("report_hash")
    frozen_hash = frozen_report.get("report_hash")
    model_hash = candidate_report.get("model_archive_hash")
    corpus_hash = frozen_report.get("corpus_archive_hash")
    if not all(
        isinstance(value, str) and _HASH.fullmatch(value)
        for value in (candidate_hash, frozen_hash, model_hash, corpus_hash)
    ):
        raise ValueError("recovery corrective frozen exam sources are invalid")
    failure_passed = bool(
        failure_exam.get("passed") is True
        and failure_exam.get("stability_retention_passed") is True
        and failure_exam.get("route_kind") == "UNSEEN_EXACT_FAILURE_STATES"
    )
    normal_passed = bool(
        normal_exam.get("passed") is True
        and normal_exam.get("stability_retention_passed") is True
        and normal_exam.get("route_kind") == "NORMAL_PARENT_ROUTE"
    )
    strongly_bound = False
    try:
        from rosclaw_soccer.training.recovery_corrective_student import (
            validate_recovery_corrective_student_evidence,
        )

        validated_candidate = validate_recovery_corrective_student_evidence(candidate_path)
        validated_frozen = validate_recovery_corrective_student_evidence(frozen_path)
        strongly_bound = bool(
            validated_candidate.get("report_hash") == candidate_hash
            and validated_frozen.get("report_hash") == frozen_hash
        )
    except (OSError, ValueError):
        # Legacy unit fixtures and v1 evidence remain readable, but cannot claim
        # the independently recomputable v2 source binding.
        strongly_bound = False
    report: dict[str, Any] = {
        "schema_version": (
            "rosclaw_soccer.recovery_corrective_frozen_exam_evidence.v2"
            if strongly_bound
            else "rosclaw_soccer.recovery_corrective_frozen_exam_evidence.v1"
        ),
        "candidate_student_report_hash": candidate_hash,
        "candidate_student_report_file_hash": hash_bytes(candidate_path.read_bytes()),
        "candidate_model_archive_hash": model_hash,
        "frozen_benchmark_student_report_hash": frozen_hash,
        "frozen_benchmark_student_report_file_hash": hash_bytes(frozen_path.read_bytes()),
        "frozen_benchmark_corpus_hash": corpus_hash,
        "frozen_failure_state_manifest_hash": frozen_report.get("failure_state_manifest_hash"),
        **{name: candidate_report[name] for name in _STUDENT_SHARED_LINEAGE},
        "failure_state_paired_physics_exam": dict(failure_exam),
        "normal_route_paired_physics_exam": dict(normal_exam),
        "frozen_failure_retention_passed": failure_passed,
        "frozen_normal_retention_passed": normal_passed,
        "frozen_benchmark_passed": bool(failure_passed and normal_passed),
        "physics_backend": "MUJOCO_MJX",
        "claim_boundary": "FROZEN_DEVELOPMENT_REGRESSION_ONLY",
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    if strongly_bound:
        report.update(
            {
                "candidate_student_report_path": str(candidate_path),
                "frozen_benchmark_student_report_path": str(frozen_path),
                "source_binding": "ABSOLUTE_PATH_FILE_HASH_AND_VALIDATED_REPORT",
                "exam_summary_binding": "RECOMPUTED_FROM_PER_SOURCE_DIAGNOSTICS",
            }
        )
    report["report_hash"] = hash_json(report)
    _atomic_json(destination, report)
    return validate_recovery_corrective_frozen_exam_evidence(destination)


def validate_recovery_corrective_frozen_exam_evidence(path: Path) -> dict[str, Any]:
    """Fail closed on modified exams or authority claims."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery corrective frozen exam evidence is invalid")
    report_hash = payload.pop("report_hash", None)
    valid_hash = report_hash == hash_json(payload)
    payload["report_hash"] = report_hash
    failure = payload.get("failure_state_paired_physics_exam")
    normal = payload.get("normal_route_paired_physics_exam")
    failure_passed = bool(
        isinstance(failure, dict)
        and failure.get("passed") is True
        and failure.get("stability_retention_passed") is True
        and failure.get("route_kind") == "UNSEEN_EXACT_FAILURE_STATES"
    )
    normal_passed = bool(
        isinstance(normal, dict)
        and normal.get("passed") is True
        and normal.get("stability_retention_passed") is True
        and normal.get("route_kind") == "NORMAL_PARENT_ROUTE"
    )
    hashes_valid = all(
        isinstance(payload.get(name), str) and _HASH.fullmatch(str(payload[name]))
        for name in (
            "candidate_student_report_hash",
            "candidate_student_report_file_hash",
            "candidate_model_archive_hash",
            "frozen_benchmark_student_report_hash",
            "frozen_benchmark_student_report_file_hash",
            "frozen_benchmark_corpus_hash",
            "frozen_failure_state_manifest_hash",
            *_STUDENT_SHARED_LINEAGE,
        )
    )
    summary_valid = bool(
        payload.get("frozen_failure_retention_passed") is failure_passed
        and payload.get("frozen_normal_retention_passed") is normal_passed
        and payload.get("frozen_benchmark_passed") is bool(failure_passed and normal_passed)
    )
    schema_version = payload.get("schema_version")
    strong_binding_valid = schema_version == (
        "rosclaw_soccer.recovery_corrective_frozen_exam_evidence.v1"
    )
    if schema_version == "rosclaw_soccer.recovery_corrective_frozen_exam_evidence.v2":
        try:
            from rosclaw_soccer.training.recovery_corrective_student import (
                RecoveryCorrectiveStudentConfig,
                _validate_corrective_exam_source_diagnostics,
                validate_recovery_corrective_student_evidence,
            )

            candidate_path = Path(str(payload.get("candidate_student_report_path")))
            frozen_path = Path(str(payload.get("frozen_benchmark_student_report_path")))
            candidate_source = validate_recovery_corrective_student_evidence(candidate_path)
            frozen_source = validate_recovery_corrective_student_evidence(frozen_path)
            frozen_config = RecoveryCorrectiveStudentConfig(**frozen_source["config"])
            strong_binding_valid = bool(
                candidate_path.is_absolute()
                and frozen_path.is_absolute()
                and candidate_path != frozen_path
                and payload.get("source_binding") == "ABSOLUTE_PATH_FILE_HASH_AND_VALIDATED_REPORT"
                and payload.get("exam_summary_binding") == "RECOMPUTED_FROM_PER_SOURCE_DIAGNOSTICS"
                and payload.get("candidate_student_report_hash")
                == candidate_source.get("report_hash")
                and payload.get("candidate_student_report_file_hash")
                == hash_bytes(candidate_path.read_bytes())
                and payload.get("candidate_model_archive_hash")
                == candidate_source.get("model_archive_hash")
                and candidate_source.get("student_development_retained") is True
                and payload.get("frozen_benchmark_student_report_hash")
                == frozen_source.get("report_hash")
                and payload.get("frozen_benchmark_student_report_file_hash")
                == hash_bytes(frozen_path.read_bytes())
                and payload.get("frozen_benchmark_corpus_hash")
                == frozen_source.get("corpus_archive_hash")
                and payload.get("frozen_failure_state_manifest_hash")
                == frozen_source.get("failure_state_manifest_hash")
                and all(
                    payload.get(name) == candidate_source.get(name) == frozen_source.get(name)
                    for name in _STUDENT_SHARED_LINEAGE
                )
                and isinstance(failure, dict)
                and isinstance(failure.get("source_diagnostics"), dict)
                and failure.get("state_count") == frozen_source.get("holdout_source_count")
                and _validate_corrective_exam_source_diagnostics(
                    failure, config=frozen_config, normal_route=False
                )
                and isinstance(normal, dict)
                and isinstance(normal.get("source_diagnostics"), dict)
                and normal.get("state_count") == frozen_source.get("holdout_source_count")
                and _validate_corrective_exam_source_diagnostics(
                    normal, config=frozen_config, normal_route=True
                )
            )
        except (KeyError, OSError, TypeError, ValueError):
            strong_binding_valid = False
    authority_valid = bool(
        schema_version
        in {
            "rosclaw_soccer.recovery_corrective_frozen_exam_evidence.v1",
            "rosclaw_soccer.recovery_corrective_frozen_exam_evidence.v2",
        }
        and payload.get("physics_backend") == "MUJOCO_MJX"
        and payload.get("claim_boundary") == "FROZEN_DEVELOPMENT_REGRESSION_ONLY"
        and payload.get("deployment_candidate") is False
        and payload.get("promotion_eligible") is False
        and payload.get("promotion_authority") == "NONE"
        and payload.get("activation_ceiling") == "SIM_ONLY"
        and payload.get("hardware_authorized") is False
        and payload.get("hardware_command_sent") is False
    )
    if not (
        valid_hash and hashes_valid and summary_valid and strong_binding_valid and authority_valid
    ):
        raise ValueError("recovery corrective frozen exam evidence is invalid")
    return payload


__all__ = [
    "RecoveryCorrectiveScaleConfig",
    "validate_recovery_corrective_frozen_exam_evidence",
    "validate_recovery_corrective_scale_evidence",
    "write_recovery_corrective_frozen_exam_evidence",
    "write_recovery_corrective_scale_evidence",
]
