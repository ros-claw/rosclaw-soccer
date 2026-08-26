"""Independent CPU qualification for the S105 recovery-command student."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.recovery_athlete_student import (
    _FEATURE_MIRROR_SIGN,
    _HOLDOUT_LANE,
    _OUTPUT_MIRROR_SIGN,
    RecoveryAthleteStudentConfig,
    _augmented_training_set,
    _lane_metrics,
    _load_success_anchors,
    decode_recovery_athlete_command,
    load_recovery_athlete_student,
)


@dataclass(frozen=True)
class RecoveryAthleteCpuExamConfig:
    """Fail-closed CPU gates independent of the four-GPU training process."""

    perturbation_factor: int = 4
    maximum_anchor_command_mae: float = 0.012
    maximum_holdout_command_mae: float = 0.015
    maximum_perturbed_normalized_rmse: float = 0.08
    maximum_perturbed_normalized_error: float = 0.35
    maximum_symmetry_error: float = 1.0e-7
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_athlete_cpu_exam_config.v1"

    def __post_init__(self) -> None:
        if not 2 <= self.perturbation_factor <= 32:
            raise ValueError("recovery athlete CPU perturbation factor is invalid")
        thresholds = (
            self.maximum_anchor_command_mae,
            self.maximum_holdout_command_mae,
            self.maximum_perturbed_normalized_rmse,
            self.maximum_perturbed_normalized_error,
            self.maximum_symmetry_error,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
            raise ValueError("recovery athlete CPU thresholds are invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("recovery athlete CPU exam must remain SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _load_training_report(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery athlete training report must be an object")
    claimed = payload.get("report_hash")
    unhashed = dict(payload)
    unhashed.pop("report_hash", None)
    if not (
        claimed == hash_json(unhashed)
        and payload.get("schema_version") == "rosclaw_soccer.recovery_athlete_training_report.v1"
        and payload.get("fit_gate_passed") is True
        and payload.get("world_size") == 4
        and payload.get("visible_cuda_devices") == 4
        and payload.get("policy_integration_completed") is False
        and payload.get("activation_ceiling") == "SIM_ONLY"
        and payload.get("hardware_command_sent") is False
        and payload.get("commercial_use_allowed") is False
    ):
        raise ValueError("recovery athlete training report authority is invalid")
    return cast(dict[str, Any], payload)


def run_recovery_athlete_cpu_exam(
    *,
    checkpoint_path: Path,
    training_report_path: Path,
    evidence_path: Path,
    parent_evidence_path: Path,
    locomotion_policy_path: Path,
    output_path: Path,
    config: RecoveryAthleteCpuExamConfig | None = None,
) -> dict[str, Any]:
    """Replay successful anchors and unseen perturbations on CPU only."""

    import torch

    active = config or RecoveryAthleteCpuExamConfig()
    checkpoint_file = checkpoint_path.expanduser().resolve()
    training_file = training_report_path.expanduser().resolve()
    locomotion = locomotion_policy_path.expanduser().resolve()
    if not checkpoint_file.is_file() or not locomotion.is_file():
        raise FileNotFoundError("recovery athlete CPU input artifact is missing")
    training = _load_training_report(training_file)
    checkpoint_hash = hash_bytes(checkpoint_file.read_bytes())
    if training.get("checkpoint_hash") != checkpoint_hash or training.get(
        "locomotion_policy_hash"
    ) != hash_bytes(locomotion.read_bytes()):
        raise ValueError("recovery athlete CPU checkpoint binding changed")
    training_config = RecoveryAthleteStudentConfig(**training["config"])
    anchors, metadata = _load_success_anchors(
        evidence_path=evidence_path,
        parent_evidence_path=parent_evidence_path,
        config=training_config,
    )
    if metadata != training.get("source_evidence"):
        raise ValueError("recovery athlete CPU source evidence changed")
    device = torch.device("cpu")
    model, checkpoint = load_recovery_athlete_student(
        checkpoint_path=checkpoint_file,
        locomotion_policy_path=locomotion,
        device=device,
    )
    if checkpoint.get("source_evidence") != metadata:
        raise ValueError("recovery athlete checkpoint source evidence changed")
    lane_metrics = _lane_metrics(torch=torch, model=model, anchors=anchors)
    maximum_anchor_mae = max(row["command_mae"] for row in lane_metrics.values())
    holdout_mae = lane_metrics[_HOLDOUT_LANE]["command_mae"]
    perturbation_config = replace(
        training_config,
        augmentation_factor=active.perturbation_factor,
        random_seed=training_config.random_seed + 100_003,
    )
    perturbed_features, perturbed_target = _augmented_training_set(
        anchors=anchors,
        config=perturbation_config,
    )
    with torch.inference_mode():
        features = torch.as_tensor(perturbed_features, device=device)
        target = torch.as_tensor(perturbed_target, device=device)
        decoded = decode_recovery_athlete_command(
            torch=torch,
            model=model,
            features=features,
        )
        error = decoded - target
        normalized_rmse = float(torch.sqrt(torch.mean(torch.square(error))))
        maximum_error = float(torch.max(torch.abs(error)))
        mirrored = features * torch.as_tensor(_FEATURE_MIRROR_SIGN, device=device)
        output_sign = torch.as_tensor(_OUTPUT_MIRROR_SIGN, device=device)
        mirror_error = float(
            torch.max(
                torch.abs(
                    decode_recovery_athlete_command(
                        torch=torch,
                        model=model,
                        features=mirrored,
                    )
                    - decoded * output_sign
                )
            )
        )
        maximum_output = float(torch.max(torch.abs(decoded)))
    gates = {
        "all_success_anchor_mae": maximum_anchor_mae <= active.maximum_anchor_command_mae,
        "unseen_holdout_lane_mae": holdout_mae <= active.maximum_holdout_command_mae,
        "independent_perturbation_rmse": normalized_rmse
        <= active.maximum_perturbed_normalized_rmse,
        "independent_perturbation_maximum_error": maximum_error
        <= active.maximum_perturbed_normalized_error,
        "exact_bilateral_equivariance": mirror_error <= active.maximum_symmetry_error,
        "bounded_normalized_output": maximum_output <= 1.0 + 1.0e-7,
    }
    passed = bool(all(gates.values()))
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_athlete_cpu_exam.v1",
        "passed": passed,
        "promotion_status": (
            "CANDIDATE_REQUIRES_CPU_MUJOCO_INTEGRATION_EXAM" if passed else "REJECTED_DEVELOPMENT"
        ),
        "physics_integration_completed": False,
        "training_report_hash": training["report_hash"],
        "training_report_file_hash": hash_bytes(training_file.read_bytes()),
        "checkpoint_hash": checkpoint_hash,
        "locomotion_policy_hash": hash_bytes(locomotion.read_bytes()),
        "source_evidence": metadata,
        "config": asdict(active),
        "config_hash": active.config_hash,
        "gates": gates,
        "metrics": {
            "maximum_anchor_command_mae": maximum_anchor_mae,
            "holdout_command_mae": holdout_mae,
            "perturbed_normalized_rmse": normalized_rmse,
            "perturbed_maximum_normalized_error": maximum_error,
            "maximum_bilateral_symmetry_error": mirror_error,
            "maximum_absolute_normalized_output": maximum_output,
            "perturbed_sample_count": int(perturbed_features.shape[0]),
        },
        "lane_metrics": lane_metrics,
        "evaluation_device": "cpu",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "implementation_hash": _implementation_hash(),
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(output_path, report)
    return report


def validate_recovery_athlete_cpu_exam(path: Path) -> dict[str, Any]:
    """Validate the CPU exam without importing torch or rerunning inference."""

    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery athlete CPU exam must be an object")
    claimed = payload.get("report_hash")
    unhashed = dict(payload)
    unhashed.pop("report_hash", None)
    gates = payload.get("gates")
    if not (
        claimed == hash_json(unhashed)
        and payload.get("schema_version") == "rosclaw_soccer.recovery_athlete_cpu_exam.v1"
        and payload.get("passed") is True
        and payload.get("promotion_status") == "CANDIDATE_REQUIRES_CPU_MUJOCO_INTEGRATION_EXAM"
        and payload.get("physics_integration_completed") is False
        and payload.get("evaluation_device") == "cpu"
        and payload.get("activation_ceiling") == "SIM_ONLY"
        and payload.get("hardware_command_sent") is False
        and payload.get("commercial_use_allowed") is False
        and payload.get("implementation_hash") == _implementation_hash()
        and isinstance(gates, dict)
        and all(gates.values())
    ):
        raise ValueError("recovery athlete CPU exam authority is invalid")
    return cast(dict[str, Any], payload)


def _implementation_hash() -> str:
    return str(
        hash_json(
            {
                "cpu_exam": hash_bytes(Path(__file__).read_bytes()),
                "student": hash_bytes(
                    (Path(__file__).parent / "recovery_athlete_student.py").read_bytes()
                ),
            }
        )
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--parent-evidence", type=Path, required=True)
    parser.add_argument("--locomotion-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_recovery_athlete_cpu_exam(
        checkpoint_path=args.checkpoint,
        training_report_path=args.training_report,
        evidence_path=args.evidence,
        parent_evidence_path=args.parent_evidence,
        locomotion_policy_path=args.locomotion_policy,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RecoveryAthleteCpuExamConfig",
    "run_recovery_athlete_cpu_exam",
    "validate_recovery_athlete_cpu_exam",
]
