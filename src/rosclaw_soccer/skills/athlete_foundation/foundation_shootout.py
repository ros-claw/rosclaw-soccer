"""Matched physics qualification for athlete foundation policies."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json


class FoundationResultStatus(StrEnum):
    KINEMATIC_ONLY = "kinematic_only"
    PHYSICS_UNQUALIFIED = "physics_unqualified"
    PHYSICS_QUALIFIED = "physics_qualified"


@dataclass(frozen=True)
class FoundationMetrics:
    tracking_success_rate: float
    joint_rmse_rad: float
    keypoint_mpjpe_m: float
    foot_slip_mps: float
    minimum_pelvis_height_m: float
    peak_torque_fraction: float
    torque_saturation_rate: float
    p95_root_angular_speed_rad_s: float
    joint_jerk_rms_rad_s3: float
    transition_error_rad: float
    recovery_rate: float
    finite_state: bool

    def __post_init__(self) -> None:
        values = tuple(
            value
            for name, value in asdict(self).items()
            if name != "finite_state" and isinstance(value, float)
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("foundation metrics must be finite")
        for name in ("tracking_success_rate", "torque_saturation_rate", "recovery_rate"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in (
            "joint_rmse_rad",
            "keypoint_mpjpe_m",
            "foot_slip_mps",
            "peak_torque_fraction",
            "p95_root_angular_speed_rad_s",
            "joint_jerk_rms_rad_s3",
            "transition_error_rad",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class FoundationThresholds:
    minimum_tracking_success_rate: float = 0.90
    maximum_joint_rmse_rad: float = 0.35
    maximum_keypoint_mpjpe_m: float = 0.12
    maximum_foot_slip_mps: float = 0.15
    minimum_pelvis_height_m: float = 0.55
    maximum_peak_torque_fraction: float = 1.0
    maximum_torque_saturation_rate: float = 0.05
    maximum_p95_root_angular_speed_rad_s: float = 2.0
    maximum_transition_error_rad: float = 0.30
    minimum_recovery_rate: float = 0.80
    schema_version: str = "rosclaw_soccer.foundation_thresholds.v1"

    def reasons(self, metrics: FoundationMetrics) -> tuple[str, ...]:
        reasons: list[str] = []
        checks = (
            (
                metrics.tracking_success_rate < self.minimum_tracking_success_rate,
                "tracking_success_below_floor",
            ),
            (metrics.joint_rmse_rad > self.maximum_joint_rmse_rad, "joint_rmse_above_ceiling"),
            (
                metrics.keypoint_mpjpe_m > self.maximum_keypoint_mpjpe_m,
                "keypoint_mpjpe_above_ceiling",
            ),
            (metrics.foot_slip_mps > self.maximum_foot_slip_mps, "foot_slip_above_ceiling"),
            (
                metrics.minimum_pelvis_height_m < self.minimum_pelvis_height_m,
                "pelvis_height_below_floor",
            ),
            (
                metrics.peak_torque_fraction > self.maximum_peak_torque_fraction,
                "peak_torque_above_limit",
            ),
            (
                metrics.torque_saturation_rate > self.maximum_torque_saturation_rate,
                "torque_saturation_above_ceiling",
            ),
            (
                metrics.p95_root_angular_speed_rad_s
                > self.maximum_p95_root_angular_speed_rad_s,
                "root_angular_speed_above_ceiling",
            ),
            (
                metrics.transition_error_rad > self.maximum_transition_error_rad,
                "transition_error_above_ceiling",
            ),
            (metrics.recovery_rate < self.minimum_recovery_rate, "recovery_below_floor"),
            (not metrics.finite_state, "non_finite_state"),
        )
        reasons.extend(reason for failed, reason in checks if failed)
        return tuple(reasons)


@dataclass(frozen=True)
class FoundationEvaluation:
    backend_id: str
    backend_contract_hash: str
    motion_atlas_hash: str
    body_hash: str
    environment_hash: str
    seed_commitment_hash: str
    candidate_artifact_hash: str
    physics_backend: str
    episode_count: int
    metrics: FoundationMetrics
    status: FoundationResultStatus
    reasons: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.foundation_evaluation.v1"

    def __post_init__(self) -> None:
        for value in (
            self.backend_contract_hash,
            self.motion_atlas_hash,
            self.body_hash,
            self.environment_hash,
            self.seed_commitment_hash,
            self.candidate_artifact_hash,
        ):
            if not value.startswith("sha256:"):
                raise ValueError("foundation evaluation requires content hashes")
        if self.episode_count < 8:
            raise ValueError("foundation physics evaluation requires at least eight episodes")
        if self.status is FoundationResultStatus.PHYSICS_QUALIFIED and self.reasons:
            raise ValueError("qualified foundation cannot contain rejection reasons")
        if self.status is FoundationResultStatus.PHYSICS_UNQUALIFIED and not self.reasons:
            raise ValueError("unqualified foundation must disclose rejection reasons")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_command_sent:
            raise ValueError("foundation evaluation must remain SIM_ONLY")

    @property
    def evaluation_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["reasons"] = list(self.reasons)
        return value


@dataclass(frozen=True)
class FoundationShootout:
    backend_contract_hashes: tuple[tuple[str, str], ...]
    motion_atlas_hash: str
    body_hash: str
    environment_hash: str
    seed_commitment_hash: str
    thresholds: FoundationThresholds
    evaluations: tuple[FoundationEvaluation, ...] = ()
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.foundation_shootout.v1"

    def __post_init__(self) -> None:
        identifiers = tuple(name for name, _ in self.backend_contract_hashes)
        if len(identifiers) != 4 or len(set(identifiers)) != 4:
            raise ValueError("foundation shootout requires four backend contracts")
        if any(not value.startswith("sha256:") for _, value in self.backend_contract_hashes):
            raise ValueError("foundation backend contract hash is invalid")
        for value in (
            self.motion_atlas_hash,
            self.body_hash,
            self.environment_hash,
            self.seed_commitment_hash,
        ):
            if not value.startswith("sha256:"):
                raise ValueError("foundation shootout binding hash is invalid")
        if len({item.backend_id for item in self.evaluations}) != len(self.evaluations):
            raise ValueError("foundation evaluation backend identifiers must be unique")
        contracts = dict(self.backend_contract_hashes)
        for result in self.evaluations:
            if contracts.get(result.backend_id) != result.backend_contract_hash:
                raise ValueError("foundation evaluation uses an unregistered backend")
            if (
                result.motion_atlas_hash != self.motion_atlas_hash
                or result.body_hash != self.body_hash
                or result.environment_hash != self.environment_hash
                or result.seed_commitment_hash != self.seed_commitment_hash
            ):
                raise ValueError("foundation evaluation is not matched to the sealed shootout")
            expected_reasons = self.thresholds.reasons(result.metrics)
            if result.status is FoundationResultStatus.PHYSICS_QUALIFIED:
                if expected_reasons:
                    raise ValueError("foundation claims qualification despite threshold failures")
            elif (
                result.status is FoundationResultStatus.PHYSICS_UNQUALIFIED
                and result.reasons != expected_reasons
            ):
                raise ValueError("foundation rejection reasons do not match thresholds")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_command_sent:
            raise ValueError("foundation shootout must remain SIM_ONLY")

    @property
    def winner_backend_id(self) -> str | None:
        qualified = tuple(
            result
            for result in self.evaluations
            if result.status is FoundationResultStatus.PHYSICS_QUALIFIED
        )
        if not qualified:
            return None
        return min(
            qualified,
            key=lambda item: (
                -item.metrics.tracking_success_rate,
                item.metrics.joint_rmse_rad,
                item.metrics.foot_slip_mps,
                item.metrics.p95_root_angular_speed_rad_s,
            ),
        ).backend_id

    @property
    def shootout_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "backend_contract_hashes": [list(item) for item in self.backend_contract_hashes],
            "motion_atlas_hash": self.motion_atlas_hash,
            "body_hash": self.body_hash,
            "environment_hash": self.environment_hash,
            "seed_commitment_hash": self.seed_commitment_hash,
            "thresholds": asdict(self.thresholds),
            "evaluations": [item.to_dict() for item in self.evaluations],
            "winner_backend_id": self.winner_backend_id,
            "status": "WINNER_QUALIFIED" if self.winner_backend_id else "PHYSICS_EVIDENCE_PENDING",
            "activation_ceiling": self.activation_ceiling,
            "hardware_command_sent": self.hardware_command_sent,
        }
        if include_hash:
            value["shootout_hash"] = self.shootout_hash
        return value


def write_foundation_shootout(shootout: FoundationShootout, output_path: Path) -> None:
    """Atomically write a shootout report without mutating training state."""

    target = output_path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(shootout.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


__all__ = [
    "FoundationEvaluation",
    "FoundationMetrics",
    "FoundationResultStatus",
    "FoundationShootout",
    "FoundationThresholds",
    "write_foundation_shootout",
]
