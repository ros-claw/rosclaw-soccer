"""Content-bound state-wise portfolio of impact-recovery teachers."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.impact_recovery_distillation import (
    ImpactRecoveryDistillationConfig,
    _train_student,
    validate_impact_recovery_distilled_student,
)
from rosclaw_soccer.training.recovery_mjx import _atomic_json

_JOINT_COUNT = 29
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_CORPUS_NAMES = {
    "actor_observation",
    "commanded_action",
    "curriculum_index",
    "control_step",
    "accepted_state_row",
    "gated_state_accepted",
    "teacher_success",
    "teacher_maximum_stable_streak",
    "cost_improvement_fraction",
}


@dataclass(frozen=True)
class ImpactRecoveryTeacherPortfolioConfig:
    """Fail-closed portfolio coverage and authority contract."""

    minimum_source_count: int = 2
    maximum_source_count: int = 8
    minimum_union_state_gain: int = 2
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.impact_recovery_teacher_portfolio_config.v1"

    def __post_init__(self) -> None:
        if (
            not 2 <= self.minimum_source_count <= 8
            or not self.minimum_source_count <= self.maximum_source_count <= 8
            or not 1 <= self.minimum_union_state_gain <= 32
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("impact-recovery teacher portfolio config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class _PortfolioSource:
    path: Path
    report: dict[str, Any]
    arrays: dict[str, np.ndarray[Any, Any]]
    horizon_steps: int
    state_scores: dict[int, tuple[int, int, float, float]]


def _state_scores(
    arrays: dict[str, np.ndarray[Any, Any]], horizon_steps: int
) -> dict[int, tuple[int, int, float, float]]:
    variant_index = arrays["curriculum_index"][::horizon_steps]
    accepted = arrays["gated_state_accepted"].astype(np.bool_)
    scores: dict[int, tuple[int, int, float, float]] = {}
    for raw_state in np.unique(variant_index[accepted]):
        state = int(raw_state)
        mask = (variant_index == state) & accepted
        scores[state] = (
            int(np.sum(mask)),
            int(np.sum(arrays["teacher_success"][mask])),
            float(np.min(arrays["teacher_maximum_stable_streak"][mask])),
            float(np.min(arrays["cost_improvement_fraction"][mask])),
        )
    return scores


def _load_source(path: Path) -> _PortfolioSource:
    resolved = path.expanduser().resolve()
    report = validate_impact_recovery_distilled_student(resolved)
    corpus_path = resolved.parent / str(report["corpus"])
    with np.load(corpus_path, allow_pickle=False) as archive:
        if not _CORPUS_NAMES.issubset(archive.files):
            raise ValueError("impact-recovery portfolio source corpus is incomplete")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    horizon_steps = int(report["residual_authority_steps"])
    row_count = int(arrays["curriculum_index"].size)
    variant_count = int(arrays["gated_state_accepted"].size)
    variant_index = arrays["curriculum_index"][::horizon_steps]
    expected_step = np.tile(np.arange(horizon_steps, dtype=np.int32), variant_count)
    if (
        horizon_steps <= 0
        or row_count != variant_count * horizon_steps
        or arrays["actor_observation"].ndim != 2
        or arrays["actor_observation"].shape[0] != row_count
        or arrays["commanded_action"].shape != (row_count, _JOINT_COUNT)
        or arrays["curriculum_index"].shape != (row_count,)
        or arrays["control_step"].shape != (row_count,)
        or arrays["accepted_state_row"].shape != (row_count,)
        or variant_index.shape != (variant_count,)
        or any(
            arrays[name].shape != (variant_count,)
            for name in (
                "gated_state_accepted",
                "teacher_success",
                "teacher_maximum_stable_streak",
                "cost_improvement_fraction",
            )
        )
        or not np.array_equal(arrays["curriculum_index"], np.repeat(variant_index, horizon_steps))
        or not np.array_equal(arrays["control_step"], expected_step)
        or not np.array_equal(
            arrays["accepted_state_row"].astype(np.bool_),
            np.repeat(arrays["gated_state_accepted"].astype(np.bool_), horizon_steps),
        )
        or any(
            not np.all(np.isfinite(arrays[name]))
            for name in (
                "actor_observation",
                "commanded_action",
                "teacher_maximum_stable_streak",
                "cost_improvement_fraction",
            )
        )
    ):
        raise ValueError("impact-recovery portfolio source semantics changed")
    return _PortfolioSource(
        path=resolved,
        report=report,
        arrays=arrays,
        horizon_steps=horizon_steps,
        state_scores=_state_scores(arrays, horizon_steps),
    )


def _select_portfolio_rows(
    sources: tuple[_PortfolioSource, ...],
) -> tuple[dict[str, np.ndarray[Any, Any]], list[dict[str, Any]], list[int]]:
    winners: dict[int, tuple[tuple[int, int, float, float, int], int]] = {}
    source_presence: dict[int, int] = {}
    for source_index, source in enumerate(sources):
        for state, raw_score in source.state_scores.items():
            score = (*raw_score, -source_index)
            source_presence[state] = source_presence.get(state, 0) + 1
            if state not in winners or score > winners[state][0]:
                winners[state] = (score, source_index)
    if not winners:
        raise ValueError("impact-recovery portfolio has no accepted teacher states")
    output: dict[str, list[np.ndarray[Any, Any]]] = {
        "actor_observation": [],
        "commanded_action": [],
        "curriculum_index": [],
        "control_step": [],
        "source_index": [],
        "accepted_state_row": [],
    }
    selected_state_counts: list[int] = []
    for source_index, source in enumerate(sources):
        row_won = np.asarray(
            [
                winners.get(int(value), ((0, 0, 0.0, 0.0, 0), -1))[1] == source_index
                for value in source.arrays["curriculum_index"]
            ],
            dtype=np.bool_,
        )
        selected = source.arrays["accepted_state_row"].astype(np.bool_) & row_won
        selected_state_counts.append(
            int(np.unique(source.arrays["curriculum_index"][selected]).size)
        )
        for name in ("actor_observation", "commanded_action", "curriculum_index", "control_step"):
            output[name].append(source.arrays[name][selected])
        output["source_index"].append(np.full(int(np.sum(selected)), source_index, dtype=np.int16))
        output["accepted_state_row"].append(np.ones(int(np.sum(selected)), dtype=np.bool_))
    combined = {name: np.concatenate(values, axis=0) for name, values in output.items()}
    winner_rows = []
    for state, (score, source_index) in sorted(winners.items()):
        winner_rows.append(
            {
                "curriculum_index": state,
                "source_index": source_index,
                "accepted_variant_count": score[0],
                "successful_variant_count": score[1],
                "minimum_stable_streak": score[2],
                "minimum_cost_improvement_fraction": score[3],
                "source_overlap_count": source_presence[state],
            }
        )
    return combined, winner_rows, selected_state_counts


def _same_lineage(sources: tuple[_PortfolioSource, ...]) -> ImpactRecoveryDistillationConfig:
    first = sources[0].report
    config = ImpactRecoveryDistillationConfig(**cast(dict[str, Any], first["config"]))
    fields = (
        "config_hash",
        "curriculum_manifest_hash",
        "curriculum_archive_hash",
        "body_hash",
        "action_semantics",
        "residual_authority_steps",
    )
    if (
        config.student_model_type != "RIDGE_CURRENT_FRAME"
        or config.label_outcome_mode != "SUCCESSOR_FRONTIER"
        or any(
            any(source.report.get(name) != first.get(name) for name in fields) for source in sources
        )
    ):
        raise ValueError("impact-recovery portfolio source lineage or model contract changed")
    return config


def build_impact_recovery_teacher_portfolio(
    *,
    source_report_paths: tuple[Path, ...],
    output_dir: Path,
    source_checkout_path: Path,
    config: ImpactRecoveryTeacherPortfolioConfig | None = None,
) -> dict[str, Any]:
    """Select one robust teacher per state and fit a sealed-exam ridge student."""

    active = config or ImpactRecoveryTeacherPortfolioConfig()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    resolved_sources = tuple(path.expanduser().resolve() for path in source_report_paths)
    if (
        not active.minimum_source_count <= len(resolved_sources) <= active.maximum_source_count
        or len(set(resolved_sources)) != len(resolved_sources)
        or destination.exists()
        or destination == checkout
        or checkout in destination.parents
    ):
        raise ValueError("impact-recovery teacher portfolio paths are invalid")
    sources = tuple(_load_source(path) for path in resolved_sources)
    student_config = _same_lineage(sources)
    corpus, winner_rows, selected_state_counts = _select_portfolio_rows(sources)
    union_state_count = len(winner_rows)
    source_state_counts = [len(source.state_scores) for source in sources]
    union_state_gain = union_state_count - max(source_state_counts)
    if union_state_count <= student_config.holdout_state_count:
        raise ValueError("impact-recovery portfolio has too few independent states")
    model, metrics = _train_student(
        observation=corpus["actor_observation"],
        action=corpus["commanded_action"],
        state_index=corpus["curriculum_index"],
        config=student_config,
    )
    destination.mkdir(parents=True)
    corpus_path = destination / "portfolio-corpus.npz"
    temporary_corpus = destination / ".portfolio-corpus.npz.tmp"
    with temporary_corpus.open("wb") as stream:
        np.savez_compressed(stream, **corpus)  # type: ignore[arg-type]
    os.replace(temporary_corpus, corpus_path)
    model_path = destination / "portfolio-student-model.npz"
    temporary_model = destination / ".portfolio-student-model.npz.tmp"
    with temporary_model.open("wb") as stream:
        np.savez_compressed(stream, **model)  # type: ignore[arg-type]
    os.replace(temporary_model, model_path)
    source_contracts = [
        {
            "path": str(source.path),
            "report_hash": source.report["report_hash"],
            "report_file_hash": hash_bytes(source.path.read_bytes()),
            "corpus_hash": source.report["corpus_hash"],
            "student_exam_eligible": source.report["student_exam_eligible"],
            "accepted_curriculum_state_count": len(source.state_scores),
        }
        for source in sources
    ]
    validation_improvement = float(metrics["validation_loss_improvement_fraction"])
    coverage_eligible = union_state_gain >= active.minimum_union_state_gain
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_teacher_portfolio.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "student_config": asdict(student_config),
        "student_config_hash": student_config.config_hash,
        "curriculum_manifest_hash": sources[0].report["curriculum_manifest_hash"],
        "body_hash": sources[0].report["body_hash"],
        "source_reports": source_contracts,
        "selection_semantics": (
            "LEXICOGRAPHIC_ACCEPTED_COUNT_SUCCESS_COUNT_MIN_STREAK_MIN_IMPROVEMENT"
        ),
        "tie_break": "LOWEST_SOURCE_INDEX",
        "state_winners": winner_rows,
        "source_selected_state_counts": selected_state_counts,
        "source_accepted_state_counts": source_state_counts,
        "union_state_count": union_state_count,
        "maximum_source_state_count": max(source_state_counts),
        "union_state_gain": union_state_gain,
        "overlap_state_count": sum(row["source_overlap_count"] > 1 for row in winner_rows),
        "selected_row_count": int(corpus["curriculum_index"].size),
        "split_semantics": "WHOLE_CURRICULUM_STATE_TRAIN_CALIBRATION_SEALED_EXAM",
        "training_metrics": metrics,
        "coverage_eligible": coverage_eligible,
        "portfolio_exam_eligible": bool(
            coverage_eligible
            and validation_improvement
            >= student_config.required_validation_loss_improvement_fraction
        ),
        "corpus": corpus_path.name,
        "corpus_hash": hash_bytes(corpus_path.read_bytes()),
        "student_model": model_path.name,
        "student_model_hash": hash_bytes(model_path.read_bytes()),
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "claim_boundary": "Portfolio warm-start candidate; matched physics exams required",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    report_path = destination / "portfolio-report.json"
    _atomic_json(report_path, report)
    return validate_impact_recovery_teacher_portfolio(report_path)


def validate_impact_recovery_teacher_portfolio(path: Path) -> dict[str, Any]:
    """Rebuild source selection, model and eligibility from bound local evidence."""

    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("impact-recovery teacher portfolio report must be an object")
    report = cast(dict[str, Any], payload)
    declared = report.pop("report_hash", None)
    try:
        config_value = report.get("config")
        student_config_value = report.get("student_config")
        source_contracts = report.get("source_reports")
        if (
            not isinstance(config_value, dict)
            or not isinstance(student_config_value, dict)
            or not isinstance(source_contracts, list)
        ):
            raise ValueError("impact-recovery teacher portfolio report is incomplete")
        config = ImpactRecoveryTeacherPortfolioConfig(**config_value)
        student_config = ImpactRecoveryDistillationConfig(**student_config_value)
        sources: list[_PortfolioSource] = []
        source_contracts_valid = bool(
            config.minimum_source_count <= len(source_contracts) <= config.maximum_source_count
        )
        for contract in source_contracts:
            if not isinstance(contract, dict) or not isinstance(contract.get("path"), str):
                source_contracts_valid = False
                continue
            source_path = Path(contract["path"]).expanduser().resolve()
            try:
                source = _load_source(source_path)
            except (FileNotFoundError, KeyError, OSError, ValueError):
                source_contracts_valid = False
                continue
            source_contracts_valid = bool(
                source_contracts_valid
                and contract.get("report_hash") == source.report.get("report_hash")
                and contract.get("report_file_hash") == hash_bytes(source_path.read_bytes())
                and contract.get("corpus_hash") == source.report.get("corpus_hash")
                and contract.get("student_exam_eligible")
                is source.report.get("student_exam_eligible")
                and contract.get("accepted_curriculum_state_count") == len(source.state_scores)
            )
            sources.append(source)
        if len(sources) != len(source_contracts):
            source_contracts_valid = False
        derived_student_config = _same_lineage(tuple(sources)) if sources else None
        corpus, winner_rows, selected_counts = (
            _select_portfolio_rows(tuple(sources)) if sources else ({}, [], [])
        )
        corpus_path = resolved.parent / str(report.get("corpus", ""))
        model_path = resolved.parent / str(report.get("student_model", ""))
        with np.load(corpus_path, allow_pickle=False) as archive:
            stored_corpus = {name: np.asarray(archive[name]) for name in archive.files}
        with np.load(model_path, allow_pickle=False) as archive:
            stored_model = {name: np.asarray(archive[name]) for name in archive.files}
        corpus_valid = set(stored_corpus) == set(corpus) and all(
            np.array_equal(stored_corpus[name], corpus[name]) for name in corpus
        )
        model_valid = bool(
            set(stored_model) == {"input_mean", "input_std", "weight", "bias"}
            and stored_model["input_mean"].shape == (189,)
            and stored_model["input_std"].shape == (189,)
            and stored_model["weight"].shape == (189, _JOINT_COUNT)
            and stored_model["bias"].shape == (_JOINT_COUNT,)
            and np.all(stored_model["input_std"] > 0.0)
            and all(np.all(np.isfinite(value)) for value in stored_model.values())
        )
        metrics = report.get("training_metrics")
        validation_improvement = (
            metrics.get("validation_loss_improvement_fraction")
            if isinstance(metrics, dict)
            else None
        )
        source_state_counts = [len(source.state_scores) for source in sources]
        union_count = len(winner_rows)
        union_gain = union_count - max(source_state_counts)
        coverage_eligible = union_gain >= config.minimum_union_state_gain
        expected_eligible = bool(
            coverage_eligible
            and isinstance(validation_improvement, int | float)
            and not isinstance(validation_improvement, bool)
            and math.isfinite(float(validation_improvement))
            and float(validation_improvement)
            >= student_config.required_validation_loss_improvement_fraction
        )
        if (
            report.get("schema_version") != "rosclaw_soccer.impact_recovery_teacher_portfolio.v1"
            or declared != hash_json(report)
            or report.get("config_hash") != config.config_hash
            or report.get("student_config_hash") != student_config.config_hash
            or derived_student_config != student_config
            or report.get("curriculum_manifest_hash")
            != sources[0].report.get("curriculum_manifest_hash")
            or report.get("body_hash") != sources[0].report.get("body_hash")
            or not source_contracts_valid
            or not corpus_valid
            or not model_valid
            or report.get("state_winners") != winner_rows
            or report.get("source_selected_state_counts") != selected_counts
            or report.get("source_accepted_state_counts") != source_state_counts
            or report.get("union_state_count") != union_count
            or report.get("maximum_source_state_count") != max(source_state_counts)
            or report.get("union_state_gain") != union_gain
            or report.get("overlap_state_count")
            != sum(row["source_overlap_count"] > 1 for row in winner_rows)
            or report.get("selected_row_count") != int(corpus["curriculum_index"].size)
            or not isinstance(metrics, dict)
            or report.get("coverage_eligible") is not coverage_eligible
            or report.get("portfolio_exam_eligible") is not expected_eligible
            or report.get("corpus") != "portfolio-corpus.npz"
            or report.get("student_model") != "portfolio-student-model.npz"
            or report.get("corpus_hash") != hash_bytes(corpus_path.read_bytes())
            or report.get("student_model_hash") != hash_bytes(model_path.read_bytes())
            or report.get("selection_semantics")
            != "LEXICOGRAPHIC_ACCEPTED_COUNT_SUCCESS_COUNT_MIN_STREAK_MIN_IMPROVEMENT"
            or report.get("tie_break") != "LOWEST_SOURCE_INDEX"
            or report.get("deployment_candidate") is not False
            or report.get("promotion_eligible") is not False
            or report.get("promotion_authority") != "NONE"
            or report.get("activation_ceiling") != "SIM_ONLY"
            or report.get("hardware_authorized") is not False
            or report.get("hardware_command_sent") is not False
            or any(
                _SHA256.fullmatch(str(report.get(name, ""))) is None
                for name in (
                    "curriculum_manifest_hash",
                    "body_hash",
                    "corpus_hash",
                    "student_model_hash",
                )
            )
            or any(
                _SHA256.fullmatch(str(contract.get(name, ""))) is None
                for contract in source_contracts
                if isinstance(contract, dict)
                for name in ("report_hash", "report_file_hash", "corpus_hash")
            )
        ):
            raise ValueError("impact-recovery teacher portfolio authority or integrity changed")
        return report
    finally:
        if declared is not None:
            report["report_hash"] = declared


__all__ = [
    "ImpactRecoveryTeacherPortfolioConfig",
    "build_impact_recovery_teacher_portfolio",
    "validate_impact_recovery_teacher_portfolio",
]
