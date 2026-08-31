"""Fail-closed selection for learned post-impact recovery candidates."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.impact_recovery_distillation import (
    validate_impact_recovery_distilled_evaluation,
    validate_impact_recovery_distilled_student,
)
from rosclaw_soccer.training.impact_recovery_mjx import (
    validate_impact_recovery_mjx_evaluation_report,
    validate_impact_recovery_mjx_report,
)
from rosclaw_soccer.training.recovery_mjx import _atomic_json

CandidateDecision = Literal["QUALIFIED_FOR_CPU_FULL_CHAIN_EXAM", "REJECTED"]
CandidateKind = Literal["MJX_PPO", "DISTILLED_STUDENT"]

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")


@dataclass(frozen=True)
class ImpactRecoveryCandidate:
    """One trained checkpoint and its matched fixed-population examination."""

    candidate_id: str
    training_report_path: Path
    evaluation_report_path: Path
    candidate_kind: CandidateKind = "MJX_PPO"

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.candidate_id) or self.candidate_kind not in {
            "MJX_PPO",
            "DISTILLED_STUDENT",
        }:
            raise ValueError("impact-recovery candidate id is invalid")


@dataclass(frozen=True)
class ImpactRecoverySelectionConfig:
    """Minimum gain and maximum forgetting allowed before a full-chain exam."""

    required_acquisition_gain_count: int = 8
    maximum_retention_drop_count: int = 4
    minimum_population_episode_count: int = 128
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.impact_recovery_selection_config.v1"

    def __post_init__(self) -> None:
        if (
            not 1 <= self.required_acquisition_gain_count <= 1_000_000
            or not 0 <= self.maximum_retention_drop_count <= 1_000_000
            or not 16 <= self.minimum_population_episode_count <= 1_000_000
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("impact-recovery selection config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def validate_impact_recovery_memory_diagnostic(path: Path) -> dict[str, Any]:
    """Validate a fixed-policy acquisition or retention baseline diagnostic."""

    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("impact-recovery memory diagnostic must be an object")
    report = cast(dict[str, Any], payload)
    declared = report.pop("report_hash", None)
    try:
        population = report.get("population")
        bins = report.get("elapsed_bins")
        seeds = report.get("seeds")
        episode_count = report.get("episode_count")
        success_count = report.get("success_count")
        if (
            report.get("schema_version")
            != "rosclaw_soccer.impact_recovery_memory_baseline_diagnostic.v1"
            or declared != hash_json(report)
            or population not in {"ACQUISITION", "RETENTION"}
            or not isinstance(report.get("mode"), str)
            or not str(report["mode"]).strip()
            or not isinstance(report.get("num_envs"), int)
            or int(report["num_envs"]) <= 0
            or not isinstance(seeds, list)
            or len(seeds) < 2
            or len(seeds) != len(set(seeds))
            or any(not isinstance(seed, int) or seed < 0 for seed in seeds)
            or not isinstance(episode_count, int)
            or episode_count != int(report["num_envs"]) * len(seeds)
            or not isinstance(success_count, int)
            or not 0 <= success_count <= episode_count
            or not isinstance(bins, dict)
            or not bins
            or report.get("activation_ceiling") != "SIM_ONLY"
            or report.get("hardware_command_sent") is not False
            or _SHA256.fullmatch(str(report.get("curriculum_manifest_hash", ""))) is None
        ):
            raise ValueError("impact-recovery memory diagnostic integrity changed")
        bin_attempts = 0
        bin_successes = 0
        for row in bins.values():
            if not isinstance(row, dict) or set(row) != {"attempts", "successes"}:
                raise ValueError("impact-recovery memory diagnostic bins are invalid")
            attempts = row["attempts"]
            successes = row["successes"]
            if (
                not isinstance(attempts, int)
                or not isinstance(successes, int)
                or not 0 <= successes <= attempts
            ):
                raise ValueError("impact-recovery memory diagnostic bin counts are invalid")
            bin_attempts += attempts
            bin_successes += successes
        if bin_attempts != episode_count or bin_successes != success_count:
            raise ValueError("impact-recovery memory diagnostic totals changed")
        return report
    finally:
        if declared is not None:
            report["report_hash"] = declared


def build_impact_recovery_selection(
    *,
    acquisition_baseline_path: Path,
    retention_baseline_path: Path,
    candidates: tuple[ImpactRecoveryCandidate, ...],
    output_dir: Path,
    source_checkout_path: Path,
    config: ImpactRecoverySelectionConfig | None = None,
) -> dict[str, Any]:
    """Select only candidates that acquire enough without forgetting memory."""

    active = config or ImpactRecoverySelectionConfig()
    destination = output_dir.expanduser().resolve()
    checkout = source_checkout_path.expanduser().resolve()
    if (
        destination.exists()
        or destination == checkout
        or checkout in destination.parents
        or destination in checkout.parents
    ):
        raise ValueError("impact-recovery selection output must be new and external")
    if not candidates or len({item.candidate_id for item in candidates}) != len(candidates):
        raise ValueError("impact-recovery selection requires unique candidates")

    acquisition_path = acquisition_baseline_path.expanduser().resolve()
    retention_path = retention_baseline_path.expanduser().resolve()
    acquisition = validate_impact_recovery_memory_diagnostic(acquisition_path)
    retention = validate_impact_recovery_memory_diagnostic(retention_path)
    if (
        acquisition.get("population") != "ACQUISITION"
        or retention.get("population") != "RETENTION"
        or acquisition.get("curriculum_manifest_hash") != retention.get("curriculum_manifest_hash")
        or acquisition.get("episode_count") != retention.get("episode_count")
        or int(acquisition["episode_count"]) < active.minimum_population_episode_count
    ):
        raise ValueError("impact-recovery baselines are not a matched fixed exam")

    baseline_episode_count = int(acquisition["episode_count"])
    baseline_acquisition = int(acquisition["success_count"])
    baseline_retention = int(retention["success_count"])
    rows: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        training_path = candidate.training_report_path.expanduser().resolve()
        evaluation_path = candidate.evaluation_report_path.expanduser().resolve()
        if candidate.candidate_kind == "DISTILLED_STUDENT":
            training = validate_impact_recovery_distilled_student(training_path)
            evaluation = validate_impact_recovery_distilled_evaluation(evaluation_path)
            report_link_valid = training.get("student_exam_eligible") is True and evaluation.get(
                "student_report_hash"
            ) == training.get("report_hash")
        else:
            training = validate_impact_recovery_mjx_report(training_path)
            evaluation = validate_impact_recovery_mjx_evaluation_report(evaluation_path)
            report_link_valid = evaluation.get("training_report_hash") == training.get(
                "report_hash"
            )
        populations = cast(dict[str, Any], evaluation["populations"])
        acquisition_row = cast(dict[str, Any], populations["acquisition"])
        retention_row = cast(dict[str, Any], populations["retention"])
        if (
            not report_link_valid
            or evaluation.get("curriculum_manifest_hash")
            != acquisition.get("curriculum_manifest_hash")
            or training.get("curriculum_manifest_hash")
            != acquisition.get("curriculum_manifest_hash")
            or acquisition_row.get("episode_count") != baseline_episode_count
            or retention_row.get("episode_count") != baseline_episode_count
        ):
            raise ValueError("impact-recovery candidate is not bound to the matched baseline")
        candidate_acquisition = int(acquisition_row["success_count"])
        candidate_retention = int(retention_row["success_count"])
        acquisition_gain = candidate_acquisition - baseline_acquisition
        retention_drop = max(0, baseline_retention - candidate_retention)
        qualified = bool(
            acquisition_gain >= active.required_acquisition_gain_count
            and retention_drop <= active.maximum_retention_drop_count
        )
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_kind": candidate.candidate_kind,
                "training_report_hash": training["report_hash"],
                "training_report_file_hash": hash_bytes(training_path.read_bytes()),
                "evaluation_report_hash": evaluation["report_hash"],
                "evaluation_report_file_hash": hash_bytes(evaluation_path.read_bytes()),
                "acquisition_success_count": candidate_acquisition,
                "acquisition_gain_count": acquisition_gain,
                "retention_success_count": candidate_retention,
                "retention_drop_count": retention_drop,
                "decision": ("QUALIFIED_FOR_CPU_FULL_CHAIN_EXAM" if qualified else "REJECTED"),
                "promotion_eligible": False,
            }
        )

    qualified_ids = [
        str(row["candidate_id"])
        for row in rows
        if row["decision"] == "QUALIFIED_FOR_CPU_FULL_CHAIN_EXAM"
    ]
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.impact_recovery_selection_report.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "curriculum_manifest_hash": acquisition["curriculum_manifest_hash"],
        "acquisition_baseline_report_hash": acquisition["report_hash"],
        "acquisition_baseline_file_hash": hash_bytes(acquisition_path.read_bytes()),
        "retention_baseline_report_hash": retention["report_hash"],
        "retention_baseline_file_hash": hash_bytes(retention_path.read_bytes()),
        "population_episode_count": baseline_episode_count,
        "baseline_acquisition_success_count": baseline_acquisition,
        "baseline_retention_success_count": baseline_retention,
        "candidates": rows,
        "qualified_candidate_ids": qualified_ids,
        "decision": (
            "CANDIDATE_READY_FOR_CPU_FULL_CHAIN_EXAM" if qualified_ids else "NO_CANDIDATE_QUALIFIED"
        ),
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "claim_boundary": (
            "Fixed MJX preselection only; CPU-MuJoCo full-chain exam is still required"
        ),
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    destination.mkdir(parents=True)
    report_path = destination / "selection-report.json"
    _atomic_json(report_path, report)
    return validate_impact_recovery_selection_report(report_path)


def validate_impact_recovery_selection_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("impact-recovery selection report must be an object")
    report = cast(dict[str, Any], payload)
    declared = report.pop("report_hash", None)
    try:
        config_value = report.get("config")
        rows = report.get("candidates")
        qualified = report.get("qualified_candidate_ids")
        if not isinstance(config_value, dict) or not isinstance(rows, list) or not rows:
            raise ValueError("impact-recovery selection report is incomplete")
        config = ImpactRecoverySelectionConfig(**config_value)
        expected_qualified: list[str] = []
        identifiers: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or not _IDENTIFIER.fullmatch(
                str(row.get("candidate_id", ""))
            ):
                raise ValueError("impact-recovery selection candidate is invalid")
            candidate_id = str(row["candidate_id"])
            candidate_kind = row.get("candidate_kind", "MJX_PPO")
            identifiers.add(candidate_id)
            acquisition_gain = row.get("acquisition_gain_count")
            retention_drop = row.get("retention_drop_count")
            expected_decision: CandidateDecision = (
                "QUALIFIED_FOR_CPU_FULL_CHAIN_EXAM"
                if isinstance(acquisition_gain, int)
                and isinstance(retention_drop, int)
                and acquisition_gain >= config.required_acquisition_gain_count
                and retention_drop <= config.maximum_retention_drop_count
                else "REJECTED"
            )
            if (
                row.get("decision") != expected_decision
                or candidate_kind not in {"MJX_PPO", "DISTILLED_STUDENT"}
                or row.get("promotion_eligible") is not False
                or any(
                    _SHA256.fullmatch(str(row.get(name, ""))) is None
                    for name in (
                        "training_report_hash",
                        "training_report_file_hash",
                        "evaluation_report_hash",
                        "evaluation_report_file_hash",
                    )
                )
            ):
                raise ValueError("impact-recovery selection decision changed")
            if expected_decision == "QUALIFIED_FOR_CPU_FULL_CHAIN_EXAM":
                expected_qualified.append(candidate_id)
        if (
            len(identifiers) != len(rows)
            or qualified != expected_qualified
            or report.get("decision")
            != (
                "CANDIDATE_READY_FOR_CPU_FULL_CHAIN_EXAM"
                if expected_qualified
                else "NO_CANDIDATE_QUALIFIED"
            )
            or report.get("schema_version") != "rosclaw_soccer.impact_recovery_selection_report.v1"
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
                    "curriculum_manifest_hash",
                    "acquisition_baseline_report_hash",
                    "acquisition_baseline_file_hash",
                    "retention_baseline_report_hash",
                    "retention_baseline_file_hash",
                )
            )
        ):
            raise ValueError("impact-recovery selection report authority or integrity changed")
        return report
    finally:
        if declared is not None:
            report["report_hash"] = declared


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-baseline", type=Path, required=True)
    parser.add_argument("--retention-baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", nargs=3, metavar=("ID", "TRAIN", "EVAL"))
    parser.add_argument(
        "--distilled-candidate",
        action="append",
        nargs=3,
        metavar=("ID", "STUDENT", "EVAL"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    args = parser.parse_args()
    candidates = tuple(
        ImpactRecoveryCandidate(name, Path(training), Path(evaluation))
        for name, training, evaluation in (args.candidate or [])
    ) + tuple(
        ImpactRecoveryCandidate(
            name,
            Path(training),
            Path(evaluation),
            candidate_kind="DISTILLED_STUDENT",
        )
        for name, training, evaluation in (args.distilled_candidate or [])
    )
    report = build_impact_recovery_selection(
        acquisition_baseline_path=args.acquisition_baseline,
        retention_baseline_path=args.retention_baseline,
        candidates=candidates,
        output_dir=args.output,
        source_checkout_path=args.source_checkout,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
