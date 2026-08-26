"""External, strict-replay evidence for the Goalkeeper V2 parent baseline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.continual.reproducibility import NumericalRuntimeContract

from rosclaw_soccer.skills.goalkeeper_v2.benchmark import (
    run_parent_coverage_time_baseline,
)
from rosclaw_soccer.skills.goalkeeper_v2.coverage_time import (
    GoalkeeperCoverageTimeReport,
    GoalkeeperCoverageTrial,
    aggregate_coverage_time,
)
from rosclaw_soccer.skills.goalkeeper_v2.policy import (
    load_goalkeeper_actor_artifact,
)
from rosclaw_soccer.skills.goalkeeper_v2.promotion import (
    GoalkeeperPromotionDecision,
    evaluate_goalkeeper_promotion,
)

_REFERENCE_COMMIT = "976a81ff19b7306bafbe923d2890066b68a85271"


@dataclass(frozen=True)
class GoalkeeperV2BaselineEvidence:
    report: GoalkeeperCoverageTimeReport
    trials: tuple[GoalkeeperCoverageTrial, ...]
    strict_replay: bool
    implementation_hash: str
    numerical_environment_passed: bool
    reference_commit: str = _REFERENCE_COMMIT
    clean_room_reimplementation: bool = True
    reference_code_copied: bool = False
    promotion_status: str = "BASELINE_ONLY_NOT_CANDIDATE"
    activation_ceiling: str = "SIM_ONLY"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_v2_baseline_evidence.v1"

    @property
    def evidence_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(include_hash=False),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            **asdict(self),
            "report": self.report.to_dict(),
            "trials": [trial.to_dict() for trial in self.trials],
            "claims": {
                "trained_goalkeeper_v2": False,
                "candidate_promoted": False,
                "coverage_time_is_measured": True,
                "actor_reads_shooter_policy_phase": False,
                "pixels_used_for_scoring": False,
                "real_hardware": False,
            },
        }
        if include_hash:
            value["evidence_hash"] = self.evidence_hash
        return value


@dataclass(frozen=True)
class GoalkeeperV2CandidateEvidence:
    parent_report: GoalkeeperCoverageTimeReport
    candidate_report: GoalkeeperCoverageTimeReport
    parent_trials: tuple[GoalkeeperCoverageTrial, ...]
    candidate_trials: tuple[GoalkeeperCoverageTrial, ...]
    promotion_decision: GoalkeeperPromotionDecision
    actor_artifact_file_hash: str
    training_report_file_hash: str
    training_report_hash: str
    strict_parent_replay: bool
    strict_candidate_replay: bool
    matched_scenario_suite: bool
    numerical_environment_passed: bool
    implementation_hash: str
    activation_ceiling: str = "SIM_ONLY"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_v2_candidate_evidence.v1"

    @property
    def evidence_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(include_hash=False),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            **asdict(self),
            "parent_report": self.parent_report.to_dict(),
            "candidate_report": self.candidate_report.to_dict(),
            "parent_trials": [trial.to_dict() for trial in self.parent_trials],
            "candidate_trials": [trial.to_dict() for trial in self.candidate_trials],
            "promotion_decision": self.promotion_decision.to_dict(),
            "claims": {
                "candidate_trained": True,
                "candidate_promoted": self.promotion_decision.verdict == "PROMOTED",
                "matched_parent_candidate_exam": self.matched_scenario_suite,
                "strict_replay": self.strict_parent_replay and self.strict_candidate_replay,
                "pixels_used_for_scoring": False,
                "real_hardware": False,
            },
        }
        if include_hash:
            value["evidence_hash"] = self.evidence_hash
        return value


def run_goalkeeper_v2_parent_evidence(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
    numerical_contract: NumericalRuntimeContract,
) -> GoalkeeperV2BaselineEvidence:
    """Run the complete 5x5 baseline twice and persist external evidence."""

    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("Goalkeeper V2 evidence must be outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    environment = numerical_contract.verify_environment()
    if not environment.passed:
        raise RuntimeError("numerical environment did not satisfy the S8 contract")
    first_report, first = run_parent_coverage_time_baseline(
        asset_root=asset_root,
        numerical_contract=numerical_contract,
    )
    _, replay = run_parent_coverage_time_baseline(
        asset_root=asset_root,
        numerical_contract=numerical_contract,
    )
    strict_replay = tuple(trial.to_dict() for trial in first) == tuple(
        trial.to_dict() for trial in replay
    )
    report = aggregate_coverage_time(
        first,
        strict_replay=strict_replay,
        sealed_holdout=False,
    )
    evidence = GoalkeeperV2BaselineEvidence(
        report=report,
        trials=first,
        strict_replay=strict_replay,
        implementation_hash=_implementation_hash(),
        numerical_environment_passed=environment.passed,
    )
    request = {
        "schema_version": "rosclaw_soccer.goalkeeper_v2_baseline_request.v1",
        "numerical_runtime_contract": numerical_contract.to_dict(),
        "numerical_runtime_contract_hash": numerical_contract.contract_hash,
        "reference": {
            "repository": "https://github.com/InternRobotics/Humanoid-Goalkeeper",
            "commit": _REFERENCE_COMMIT,
            "license": "CC-BY-NC-SA-4.0",
            "usage": "clean_room_research_reference_only",
        },
        "deadline_count": len(first_report.points),
        "trial_count": len(first),
        "promotion_requested": False,
        "activation_ceiling": "SIM_ONLY",
    }
    _write_json(output / "request.json", request)
    _write_json(output / "goalkeeper-v2-parent-baseline.json", evidence.to_dict())
    return evidence


def run_goalkeeper_v2_candidate_evidence(
    *,
    asset_root: Path,
    actor_artifact_path: Path,
    training_report_path: Path,
    output_dir: Path,
    source_checkout: Path,
    numerical_contract: NumericalRuntimeContract,
    historical_mean_regression: float | None = None,
    sealed_holdout: bool = False,
) -> GoalkeeperV2CandidateEvidence:
    """Run matched parent/candidate exams twice and persist a fail-closed decision."""

    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("Goalkeeper V2 evidence must be outside the source checkout")
    if output.exists():
        raise FileExistsError(f"candidate evidence output already exists: {output}")
    environment = numerical_contract.verify_environment()
    if not environment.passed:
        raise RuntimeError("numerical environment did not satisfy the S9 contract")
    artifact = load_goalkeeper_actor_artifact(actor_artifact_path)
    training_payload = json.loads(training_report_path.read_text(encoding="utf-8"))
    if training_payload.get("candidate_policy_hash") != artifact.policy_hash:
        raise ValueError("training report does not bind the supplied candidate actor")
    if training_payload.get("parent_policy_hash") != artifact.parent_policy_hash:
        raise ValueError("training report changed the candidate parent")

    parent_first, parent_trials = run_parent_coverage_time_baseline(
        asset_root=asset_root,
        numerical_contract=numerical_contract,
    )
    _, parent_replay = run_parent_coverage_time_baseline(
        asset_root=asset_root,
        numerical_contract=numerical_contract,
    )
    candidate_first, candidate_trials = run_parent_coverage_time_baseline(
        asset_root=asset_root,
        numerical_contract=numerical_contract,
        actor_artifact_path=actor_artifact_path,
    )
    _, candidate_replay = run_parent_coverage_time_baseline(
        asset_root=asset_root,
        numerical_contract=numerical_contract,
        actor_artifact_path=actor_artifact_path,
    )
    strict_parent = _trials_equal(parent_trials, parent_replay)
    strict_candidate = _trials_equal(candidate_trials, candidate_replay)
    parent_report = aggregate_coverage_time(
        parent_trials,
        strict_replay=strict_parent,
        sealed_holdout=sealed_holdout,
    )
    candidate_report = aggregate_coverage_time(
        candidate_trials,
        strict_replay=strict_candidate,
        sealed_holdout=sealed_holdout,
    )
    matched = parent_report.scenario_suite_hash == candidate_report.scenario_suite_hash
    decision = evaluate_goalkeeper_promotion(
        parent_report=parent_report,
        parent_trials=parent_trials,
        candidate_report=candidate_report,
        candidate_trials=candidate_trials,
        historical_mean_regression=historical_mean_regression,
    )
    evidence = GoalkeeperV2CandidateEvidence(
        parent_report=parent_report,
        candidate_report=candidate_report,
        parent_trials=parent_trials,
        candidate_trials=candidate_trials,
        promotion_decision=decision,
        actor_artifact_file_hash=_file_hash(actor_artifact_path),
        training_report_file_hash=_file_hash(training_report_path),
        training_report_hash=str(training_payload["report_hash"]),
        strict_parent_replay=strict_parent,
        strict_candidate_replay=strict_candidate,
        matched_scenario_suite=matched,
        numerical_environment_passed=environment.passed,
        implementation_hash=goalkeeper_v2_implementation_hash(),
    )
    output.mkdir(parents=True, exist_ok=False)
    _write_json(
        output / "request.json",
        {
            "schema_version": "rosclaw_soccer.goalkeeper_v2_candidate_request.v1",
            "actor_policy_hash": artifact.policy_hash,
            "parent_policy_hash": artifact.parent_policy_hash,
            "training_report_hash": training_payload["report_hash"],
            "numerical_runtime_contract": numerical_contract.to_dict(),
            "numerical_runtime_contract_hash": numerical_contract.contract_hash,
            "sealed_holdout": sealed_holdout,
            "historical_mean_regression": historical_mean_regression,
            "implementation_hash": evidence.implementation_hash,
            "activation_ceiling": "SIM_ONLY",
        },
    )
    _write_json(output / "goalkeeper-v2-candidate-evidence.json", evidence.to_dict())
    return evidence


def goalkeeper_v2_implementation_hash() -> str:
    """Bind evidence and videos to the complete CPU scoring implementation."""

    digest = hashlib.sha256()
    # This module defines the hash procedure itself.  Hashing its raw bytes
    # would make the digest self-referential and would invalidate an evidence
    # file as soon as its own hash field was serialized.  Bind the stable
    # schema tag here and all behavior-bearing downstream implementation
    # files below.
    digest.update(b"rosclaw_soccer.goalkeeper_v2_implementation.v1")
    files = (
        Path(__file__).parents[1] / "skills" / "goalkeeper_v2" / "benchmark.py",
        Path(__file__).parents[1] / "skills" / "goalkeeper_v2" / "coverage_time.py",
        Path(__file__).parents[1] / "skills" / "goalkeeper_v2" / "observations.py",
        Path(__file__).parents[1] / "skills" / "goalkeeper_v2" / "policy.py",
        Path(__file__).parents[1] / "skills" / "team" / "agility_profiler.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
        Path(__file__).parents[1] / "world" / "field.py",
    )
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _implementation_hash() -> str:
    return goalkeeper_v2_implementation_hash()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _trials_equal(
    first: tuple[GoalkeeperCoverageTrial, ...],
    second: tuple[GoalkeeperCoverageTrial, ...],
) -> bool:
    return tuple(trial.to_dict() for trial in first) == tuple(trial.to_dict() for trial in second)


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "GoalkeeperV2BaselineEvidence",
    "GoalkeeperV2CandidateEvidence",
    "goalkeeper_v2_implementation_hash",
    "run_goalkeeper_v2_candidate_evidence",
    "run_goalkeeper_v2_parent_evidence",
]
