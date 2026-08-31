"""Content-bound failure-frontier sampling for impact-recovery learning.

The frontier is derived only from an independent, uniformly sampled evaluation.
It may influence future simulator resets, but it carries no promotion or hardware
authority.  Row weights mix capability-frontier, observed-failure, and uniform
anchor distributions so that difficult states receive attention without making
the next evaluation circular.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from rosclaw.continual.failure_curriculum import (
    CapabilityBin,
    CapabilityFrontierScheduler,
    CurriculumMixture,
)

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.impact_recovery_curriculum import (
    validate_impact_recovery_curriculum,
)
from rosclaw_soccer.training.impact_recovery_mjx import (
    validate_impact_recovery_mjx_evaluation_report,
)
from rosclaw_soccer.training.recovery_mjx import _atomic_json

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class ImpactRecoveryFrontierConfig:
    """Fail-closed reset weighting derived from measured capability bins."""

    elapsed_bin_width_sec: float = 1.0
    frontier_fraction: float = 0.45
    recent_failure_fraction: float = 0.40
    uniform_anchor_fraction: float = 0.15
    lower_success: float = 0.30
    upper_success: float = 0.70
    minimum_bin_probability: float = 0.02
    minimum_attempts: int = 16
    temperature: float = 0.15
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.impact_recovery_frontier_config.v1"

    def __post_init__(self) -> None:
        finite = (
            self.elapsed_bin_width_sec,
            self.frontier_fraction,
            self.recent_failure_fraction,
            self.uniform_anchor_fraction,
            self.lower_success,
            self.upper_success,
            self.minimum_bin_probability,
            self.temperature,
        )
        if (
            any(not math.isfinite(value) for value in finite)
            or not 0.25 <= self.elapsed_bin_width_sec <= 2.0
            or any(
                value < 0.0
                for value in (
                    self.frontier_fraction,
                    self.recent_failure_fraction,
                    self.uniform_anchor_fraction,
                )
            )
            or not math.isclose(
                self.frontier_fraction
                + self.recent_failure_fraction
                + self.uniform_anchor_fraction,
                1.0,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or not 0.0 <= self.lower_success < self.upper_success <= 1.0
            or not 0.0 < self.minimum_bin_probability < 1.0
            or not 1 <= self.minimum_attempts <= 10_000
            or not 0.01 <= self.temperature <= 1.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("impact-recovery frontier config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _bin_id(elapsed_sec: float, width_sec: float) -> str:
    index = max(0, int(math.floor((elapsed_sec + 1.0e-9) / width_sec)))
    return f"elapsed_{index:03d}"


def _normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("impact-recovery frontier distribution is empty")
    return {key: value / total for key, value in values.items()}


def build_impact_recovery_frontier(
    *,
    curriculum_manifest_path: Path,
    evaluation_report_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    config: ImpactRecoveryFrontierConfig | None = None,
) -> dict[str, Any]:
    """Build a training-only frontier from independent uniform evaluation."""

    active = config or ImpactRecoveryFrontierConfig()
    curriculum_path = curriculum_manifest_path.expanduser().resolve()
    evaluation_path = evaluation_report_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if not curriculum_path.is_file() or not evaluation_path.is_file():
        raise FileNotFoundError("impact-recovery frontier inputs are incomplete")
    if (
        destination.exists()
        or destination == checkout
        or checkout in destination.parents
        or destination in checkout.parents
    ):
        raise ValueError("impact-recovery frontier output must be new and external")

    curriculum = validate_impact_recovery_curriculum(curriculum_path)
    evaluation = validate_impact_recovery_mjx_evaluation_report(evaluation_path)
    if evaluation.get("curriculum_manifest_hash") != curriculum.get("manifest_hash"):
        raise ValueError("impact-recovery frontier curriculum binding changed")
    populations = cast(dict[str, Any], evaluation["populations"])
    acquisition = cast(dict[str, Any], populations["acquisition"])
    curriculum_rows = cast(list[dict[str, Any]], curriculum["rows"])
    acquisition_rows = {
        int(row["archive_row"]): row
        for row in curriculum_rows
        if row.get("source_use") == "ACQUISITION_FAILURE" and row.get("succeeded") is False
    }
    if not acquisition_rows:
        raise ValueError("impact-recovery frontier has no acquisition rows")

    observations: dict[int, list[bool]] = defaultdict(list)
    for repeat in cast(list[Any], acquisition.get("repeats")):
        if not isinstance(repeat, dict) or not isinstance(repeat.get("episode_metrics"), dict):
            raise ValueError("impact-recovery frontier evaluation metrics are incomplete")
        metrics = cast(dict[str, Any], repeat["episode_metrics"])
        row_values = metrics.get("curriculum_row_once")
        success_values = metrics.get("success")
        elapsed_values = metrics.get("elapsed_since_contact_once")
        if (
            not isinstance(row_values, list)
            or not isinstance(success_values, list)
            or not isinstance(elapsed_values, list)
            or not row_values
            or len(row_values) != len(success_values)
            or len(row_values) != len(elapsed_values)
        ):
            raise ValueError("impact-recovery frontier episode arrays are invalid")
        for raw_row, raw_success, raw_elapsed in zip(
            row_values, success_values, elapsed_values, strict=True
        ):
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (raw_row, raw_success, raw_elapsed)
            ):
                raise ValueError("impact-recovery frontier metrics must be finite")
            archive_row = int(round(float(raw_row))) - 1
            source_row = acquisition_rows.get(archive_row)
            if (
                source_row is None
                or not math.isclose(float(raw_row), archive_row + 1.0, abs_tol=1.0e-5)
                or not math.isclose(
                    float(raw_elapsed),
                    float(source_row["elapsed_since_contact_sec"]),
                    abs_tol=1.0e-4,
                )
                or float(raw_success) not in {0.0, 1.0}
            ):
                raise ValueError("impact-recovery frontier row identity changed")
            observations[archive_row].append(float(raw_success) == 1.0)

    bin_rows: dict[str, list[int]] = defaultdict(list)
    bin_attempts: dict[str, int] = defaultdict(int)
    bin_successes: dict[str, int] = defaultdict(int)
    for archive_row, row in acquisition_rows.items():
        bin_name = _bin_id(float(row["elapsed_since_contact_sec"]), active.elapsed_bin_width_sec)
        bin_rows[bin_name].append(archive_row)
        results = observations.get(archive_row, [])
        bin_attempts[bin_name] += len(results)
        bin_successes[bin_name] += sum(results)

    maximum_elapsed = max(
        float(row["elapsed_since_contact_sec"]) for row in acquisition_rows.values()
    )
    capability_bins = tuple(
        CapabilityBin(
            bin_id=bin_name,
            difficulty=max(
                0.0,
                min(
                    1.0,
                    1.0
                    - min(float(acquisition_rows[row]["elapsed_since_contact_sec"]) for row in rows)
                    / maximum_elapsed,
                ),
            ),
            successes=bin_successes[bin_name],
            attempts=bin_attempts[bin_name],
        )
        for bin_name, rows in sorted(bin_rows.items())
    )
    scheduler = CapabilityFrontierScheduler(
        lower_success=active.lower_success,
        upper_success=active.upper_success,
        minimum_probability=active.minimum_bin_probability,
        minimum_attempts=active.minimum_attempts,
        temperature=active.temperature,
    )
    frontier_probability = dict(scheduler.probabilities(capability_bins))
    failure_score = _normalize(
        {
            item.bin_id: (item.attempts - item.successes + 1.0) / (item.attempts + 2.0)
            for item in capability_bins
        }
    )
    uniform_probability = 1.0 / len(capability_bins)
    bin_probability = _normalize(
        {
            item.bin_id: (
                active.frontier_fraction * frontier_probability[item.bin_id]
                + active.recent_failure_fraction * failure_score[item.bin_id]
                + active.uniform_anchor_fraction * uniform_probability
            )
            for item in capability_bins
        }
    )
    mixture = CurriculumMixture(
        capability_frontier=active.frontier_fraction,
        recent_failure=active.recent_failure_fraction,
        historical_anchor=active.uniform_anchor_fraction,
        nightmare=0.0,
        social_teacher=0.0,
    )

    rows: list[dict[str, Any]] = []
    for archive_row, source_row in sorted(acquisition_rows.items()):
        bin_name = _bin_id(
            float(source_row["elapsed_since_contact_sec"]), active.elapsed_bin_width_sec
        )
        results = observations.get(archive_row, [])
        rows.append(
            {
                "archive_row": archive_row,
                "bin_id": bin_name,
                "elapsed_since_contact_sec": float(source_row["elapsed_since_contact_sec"]),
                "attempts": len(results),
                "successes": sum(results),
                "success_rate": None if not results else sum(results) / len(results),
                "sampling_probability": bin_probability[bin_name] / len(bin_rows[bin_name]),
            }
        )

    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_frontier.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "curriculum_manifest_hash": curriculum["manifest_hash"],
        "source_evaluation_report_hash": evaluation["report_hash"],
        "source_checkpoint_hash": evaluation["selected_checkpoint_hash"],
        "evaluation_sampling_semantics": "INDEPENDENT_UNIFORM_ACQUISITION",
        "mixture": {source.value: weight for source, weight in mixture.as_mapping().items()},
        "bins": [
            {
                "bin_id": item.bin_id,
                "difficulty": item.difficulty,
                "attempts": item.attempts,
                "successes": item.successes,
                "success_rate": item.success_rate,
                "sampling_probability": bin_probability[item.bin_id],
            }
            for item in capability_bins
        ],
        "rows": rows,
        "row_count": len(rows),
        "observed_episode_count": sum(len(value) for value in observations.values()),
        "training_use_only": True,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["manifest_hash"] = hash_json(report)
    destination.mkdir(parents=True)
    path = destination / "impact-recovery-frontier.json"
    _atomic_json(path, report)
    return validate_impact_recovery_frontier(path)


def validate_impact_recovery_frontier(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("impact-recovery frontier must be an object")
    report = cast(dict[str, Any], payload)
    declared = report.pop("manifest_hash", None)
    try:
        config_value = report.get("config")
        rows = report.get("rows")
        bins = report.get("bins")
        if (
            not isinstance(config_value, dict)
            or not isinstance(rows, list)
            or not isinstance(bins, list)
        ):
            raise ValueError("impact-recovery frontier is incomplete")
        config = ImpactRecoveryFrontierConfig(**config_value)
        archive_rows: set[int] = set()
        bin_ids: set[str] = set()
        bin_probability = 0.0
        bin_attempts = 0
        for item in bins:
            if not isinstance(item, dict) or set(item) != {
                "bin_id",
                "difficulty",
                "attempts",
                "successes",
                "success_rate",
                "sampling_probability",
            }:
                raise ValueError("impact-recovery frontier bin is invalid")
            bin_id = item.get("bin_id")
            difficulty = item.get("difficulty")
            attempts = item.get("attempts")
            successes = item.get("successes")
            item_probability = item.get("sampling_probability")
            if (
                not isinstance(bin_id, str)
                or not bin_id
                or bin_id in bin_ids
                or not isinstance(difficulty, (int, float))
                or not math.isfinite(float(difficulty))
                or not 0.0 <= float(difficulty) <= 1.0
                or isinstance(attempts, bool)
                or not isinstance(attempts, int)
                or attempts < 0
                or isinstance(successes, bool)
                or not isinstance(successes, int)
                or not 0 <= successes <= attempts
                or item.get("success_rate") != (None if attempts == 0 else successes / attempts)
                or not isinstance(item_probability, (int, float))
                or not math.isfinite(float(item_probability))
                or float(item_probability) <= 0.0
            ):
                raise ValueError("impact-recovery frontier bin statistics are invalid")
            bin_ids.add(bin_id)
            bin_probability += float(item_probability)
            bin_attempts += attempts
        probability = 0.0
        row_attempts = 0
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "archive_row",
                "bin_id",
                "elapsed_since_contact_sec",
                "attempts",
                "successes",
                "success_rate",
                "sampling_probability",
            }:
                raise ValueError("impact-recovery frontier row is invalid")
            archive_row = row.get("archive_row")
            attempts = row.get("attempts")
            successes = row.get("successes")
            row_probability = row.get("sampling_probability")
            elapsed = row.get("elapsed_since_contact_sec")
            if (
                isinstance(archive_row, bool)
                or not isinstance(archive_row, int)
                or archive_row < 0
                or archive_row in archive_rows
                or not isinstance(row.get("bin_id"), str)
                or row.get("bin_id") not in bin_ids
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or float(elapsed) < 0.0
                or isinstance(attempts, bool)
                or not isinstance(attempts, int)
                or attempts < 0
                or isinstance(successes, bool)
                or not isinstance(successes, int)
                or not 0 <= successes <= attempts
                or not isinstance(row_probability, (int, float))
                or not math.isfinite(float(row_probability))
                or float(row_probability) <= 0.0
                or row.get("success_rate") != (None if attempts == 0 else successes / attempts)
            ):
                raise ValueError("impact-recovery frontier row statistics are invalid")
            archive_rows.add(archive_row)
            probability += float(row_probability)
            row_attempts += attempts
        expected_mixture = CurriculumMixture(
            capability_frontier=config.frontier_fraction,
            recent_failure=config.recent_failure_fraction,
            historical_anchor=config.uniform_anchor_fraction,
            nightmare=0.0,
            social_teacher=0.0,
        )
        expected_mixture_value = {
            source.value: weight for source, weight in expected_mixture.as_mapping().items()
        }
        if (
            report.get("schema_version") != "rosclaw_soccer.impact_recovery_frontier.v1"
            or declared != hash_json(report)
            or report.get("config_hash") != config.config_hash
            or report.get("row_count") != len(rows)
            or not rows
            or not bins
            or not math.isclose(probability, 1.0, rel_tol=0.0, abs_tol=1.0e-9)
            or not math.isclose(bin_probability, 1.0, rel_tol=0.0, abs_tol=1.0e-9)
            or report.get("observed_episode_count") != row_attempts
            or bin_attempts != row_attempts
            or report.get("mixture") != expected_mixture_value
            or report.get("evaluation_sampling_semantics") != "INDEPENDENT_UNIFORM_ACQUISITION"
            or report.get("training_use_only") is not True
            or report.get("promotion_eligible") is not False
            or report.get("promotion_authority") != "NONE"
            or report.get("activation_ceiling") != "SIM_ONLY"
            or report.get("hardware_authorized") is not False
            or report.get("hardware_command_sent") is not False
            or any(
                _SHA256.fullmatch(str(report.get(name, ""))) is None
                for name in (
                    "curriculum_manifest_hash",
                    "source_evaluation_report_hash",
                    "source_checkpoint_hash",
                )
            )
        ):
            raise ValueError("impact-recovery frontier authority or integrity changed")
        return report
    finally:
        if declared is not None:
            report["manifest_hash"] = declared


__all__ = [
    "ImpactRecoveryFrontierConfig",
    "build_impact_recovery_frontier",
    "validate_impact_recovery_frontier",
]
