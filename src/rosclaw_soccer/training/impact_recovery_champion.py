"""Content-bound incumbent/challenger comparison for recovery candidates."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from rosclaw.continual.champion_registry import (
    DominanceMetricRole,
    PairedDominanceEvidence,
    PairedDominanceMetric,
)

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.impact_recovery_mjx import (
    validate_impact_recovery_mjx_evaluation_report,
    validate_impact_recovery_mjx_report,
)
from rosclaw_soccer.training.recovery_mjx import _atomic_json

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class ImpactRecoveryChampionChallengeConfig:
    """Strict growth and retention tolerances for one paired GPU challenge."""

    minimum_acquisition_gain_count: int = 1
    maximum_retention_drop_count: int = 1
    minimum_population_episode_count: int = 128
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.impact_recovery_champion_challenge_config.v1"

    def __post_init__(self) -> None:
        if (
            not 1 <= self.minimum_acquisition_gain_count <= 1_000_000
            or not 0 <= self.maximum_retention_drop_count <= 1_000_000
            or not 16 <= self.minimum_population_episode_count <= 1_000_000
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("impact-recovery champion challenge config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _population_successes(report: dict[str, Any], name: str) -> tuple[int, int]:
    populations = cast(dict[str, Any], report["populations"])
    population = cast(dict[str, Any], populations[name])
    return int(population["success_count"]), int(population["episode_count"])


def _checkpoint_manifest_bound(
    evaluation: dict[str, Any],
    training: dict[str, Any],
) -> bool:
    """Require the evaluated checkpoint manifest to exist in its training tree."""

    selected = evaluation.get("selected_checkpoint_files")
    training_files = training.get("checkpoint_files")
    if not isinstance(selected, list) or not isinstance(training_files, list):
        return False
    prefixes = {
        str(row["path"]).split("/", maxsplit=1)[0]
        for row in training_files
        if isinstance(row, dict) and "/" in str(row.get("path", ""))
    }
    for prefix in prefixes:
        marker = f"{prefix}/"
        normalized = [
            {**row, "path": str(row["path"])[len(marker) :]}
            for row in training_files
            if isinstance(row, dict) and str(row.get("path", "")).startswith(marker)
        ]
        if normalized == selected:
            return True
    return False


def build_impact_recovery_champion_challenge(
    *,
    incumbent_training_report_path: Path,
    incumbent_evaluation_report_path: Path,
    challenger_training_report_path: Path,
    challenger_evaluation_report_path: Path,
    output_dir: Path,
    source_checkout_path: Path,
    config: ImpactRecoveryChampionChallengeConfig | None = None,
) -> dict[str, Any]:
    """Compare two checkpoints on an identical suite without granting promotion."""

    active = config or ImpactRecoveryChampionChallengeConfig()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if (
        destination.exists()
        or destination == checkout
        or checkout in destination.parents
        or destination in checkout.parents
    ):
        raise ValueError("impact-recovery champion challenge output must be new and external")
    incumbent_training_path = incumbent_training_report_path.expanduser().resolve()
    incumbent_evaluation_path = incumbent_evaluation_report_path.expanduser().resolve()
    challenger_training_path = challenger_training_report_path.expanduser().resolve()
    challenger_evaluation_path = challenger_evaluation_report_path.expanduser().resolve()
    incumbent_training = validate_impact_recovery_mjx_report(incumbent_training_path)
    incumbent = validate_impact_recovery_mjx_evaluation_report(incumbent_evaluation_path)
    challenger_training = validate_impact_recovery_mjx_report(challenger_training_path)
    challenger = validate_impact_recovery_mjx_evaluation_report(challenger_evaluation_path)
    incumbent_acquisition, incumbent_episodes = _population_successes(incumbent, "acquisition")
    challenger_acquisition, challenger_episodes = _population_successes(challenger, "acquisition")
    incumbent_retention, incumbent_retention_episodes = _population_successes(
        incumbent, "retention"
    )
    challenger_retention, challenger_retention_episodes = _population_successes(
        challenger, "retention"
    )
    if (
        incumbent.get("training_report_hash") != incumbent_training.get("report_hash")
        or challenger.get("training_report_hash") != challenger_training.get("report_hash")
        or not _checkpoint_manifest_bound(incumbent, incumbent_training)
        or not _checkpoint_manifest_bound(challenger, challenger_training)
        or incumbent.get("curriculum_manifest_hash") != challenger.get("curriculum_manifest_hash")
        or incumbent.get("config_hash") != challenger.get("config_hash")
        or len(
            {
                incumbent_episodes,
                challenger_episodes,
                incumbent_retention_episodes,
                challenger_retention_episodes,
            }
        )
        != 1
        or incumbent_episodes < active.minimum_population_episode_count
    ):
        raise ValueError("impact-recovery champion challenge suite changed")
    scenario_suite_hash = hash_json(
        {
            "curriculum_manifest_hash": incumbent["curriculum_manifest_hash"],
            "evaluation_config_hash": incumbent["config_hash"],
            "population_episode_count": incumbent_episodes,
            "populations": ["acquisition", "retention"],
        }
    )
    dominance = PairedDominanceEvidence(
        incumbent_artifact_hash=str(incumbent["selected_checkpoint_hash"]),
        challenger_artifact_hash=str(challenger["selected_checkpoint_hash"]),
        scenario_suite_hash=str(scenario_suite_hash),
        metrics=(
            PairedDominanceMetric(
                metric_id="acquisition_success_count",
                incumbent_value=float(incumbent_acquisition),
                challenger_value=float(challenger_acquisition),
                higher_is_better=True,
                role=DominanceMetricRole.OBJECTIVE,
                minimum_improvement=float(active.minimum_acquisition_gain_count),
            ),
            PairedDominanceMetric(
                metric_id="retention_success_count",
                incumbent_value=float(incumbent_retention),
                challenger_value=float(challenger_retention),
                higher_is_better=True,
                role=DominanceMetricRole.GUARDRAIL,
                maximum_regression=float(active.maximum_retention_drop_count),
            ),
        ),
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_champion_challenge.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "incumbent_training_report_hash": incumbent_training["report_hash"],
        "incumbent_training_report_file_hash": hash_bytes(incumbent_training_path.read_bytes()),
        "incumbent_evaluation_report_hash": incumbent["report_hash"],
        "incumbent_evaluation_report_file_hash": hash_bytes(incumbent_evaluation_path.read_bytes()),
        "challenger_training_report_hash": challenger_training["report_hash"],
        "challenger_training_report_file_hash": hash_bytes(challenger_training_path.read_bytes()),
        "challenger_evaluation_report_hash": challenger["report_hash"],
        "challenger_evaluation_report_file_hash": hash_bytes(
            challenger_evaluation_path.read_bytes()
        ),
        "scenario_suite_hash": scenario_suite_hash,
        "dominance_evidence": dominance.to_dict(),
        "dominance_evidence_hash": dominance.evidence_hash,
        "decision": (
            "CHALLENGER_READY_FOR_CPU_FULL_CHAIN_EXAM"
            if dominance.promotion_passed
            else "CHALLENGER_ARCHIVED"
        ),
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "claim_boundary": "Paired GPU preselection only; CPU-MuJoCo full-chain exam required",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    destination.mkdir(parents=True)
    path = destination / "champion-challenge.json"
    _atomic_json(path, report)
    return validate_impact_recovery_champion_challenge(path)


def validate_impact_recovery_champion_challenge(path: Path) -> dict[str, Any]:
    """Recompute the generic paired-dominance decision from a saved report."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("impact-recovery champion challenge report is invalid")
    report = cast(dict[str, Any], payload)
    declared = report.pop("report_hash", None)
    try:
        config_value = report.get("config")
        evidence_value = report.get("dominance_evidence")
        if not isinstance(config_value, dict) or not isinstance(evidence_value, dict):
            raise ValueError("impact-recovery champion challenge report is incomplete")
        config = ImpactRecoveryChampionChallengeConfig(**config_value)
        metric_values = evidence_value.get("metrics")
        if not isinstance(metric_values, list):
            raise ValueError("impact-recovery champion challenge metrics are missing")
        metrics = tuple(
            PairedDominanceMetric(
                metric_id=str(item["metric_id"]),
                incumbent_value=float(item["incumbent_value"]),
                challenger_value=float(item["challenger_value"]),
                higher_is_better=bool(item["higher_is_better"]),
                role=DominanceMetricRole(str(item["role"])),
                minimum_improvement=float(item["minimum_improvement"]),
                maximum_regression=float(item["maximum_regression"]),
            )
            for item in metric_values
            if isinstance(item, dict)
        )
        dominance = PairedDominanceEvidence(
            incumbent_artifact_hash=str(evidence_value.get("incumbent_artifact_hash", "")),
            challenger_artifact_hash=str(evidence_value.get("challenger_artifact_hash", "")),
            scenario_suite_hash=str(evidence_value.get("scenario_suite_hash", "")),
            metrics=metrics,
            evidence_domain=str(evidence_value.get("evidence_domain", "")),
        )
        expected_decision = (
            "CHALLENGER_READY_FOR_CPU_FULL_CHAIN_EXAM"
            if dominance.promotion_passed
            else "CHALLENGER_ARCHIVED"
        )
        if (
            len(metrics) != len(metric_values)
            or evidence_value != dominance.to_dict()
            or report.get("dominance_evidence_hash") != dominance.evidence_hash
            or report.get("scenario_suite_hash") != dominance.scenario_suite_hash
            or report.get("decision") != expected_decision
            or report.get("schema_version")
            != "rosclaw_soccer.impact_recovery_champion_challenge.v1"
            or declared != hash_json(report)
            or report.get("config_hash") != config.config_hash
            or report.get("promotion_eligible") is not False
            or report.get("promotion_authority") != "NONE"
            or report.get("activation_ceiling") != "SIM_ONLY"
            or report.get("hardware_authorized") is not False
            or report.get("hardware_command_sent") is not False
            or any(
                _SHA256.fullmatch(str(report.get(name, ""))) is None
                for name in (
                    "incumbent_training_report_hash",
                    "incumbent_training_report_file_hash",
                    "incumbent_evaluation_report_hash",
                    "incumbent_evaluation_report_file_hash",
                    "challenger_training_report_hash",
                    "challenger_training_report_file_hash",
                    "challenger_evaluation_report_hash",
                    "challenger_evaluation_report_file_hash",
                    "scenario_suite_hash",
                    "dominance_evidence_hash",
                )
            )
        ):
            raise ValueError("impact-recovery champion challenge integrity changed")
        return report
    finally:
        if declared is not None:
            report["report_hash"] = declared


__all__ = [
    "ImpactRecoveryChampionChallengeConfig",
    "build_impact_recovery_champion_challenge",
    "validate_impact_recovery_champion_challenge",
]
