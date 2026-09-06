"""Data-driven residual calibration between tactical delivery and physical aim.

The frozen whole-body kick prior can have a systematic delivery bias.  This
module learns that bias from physics-scored attempts instead of hiding it in a
scenario script.  The resulting actor only proposes a world-frame aim point;
it has no joint, torque, ROS, or hardware authority.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class PassAimCalibrationSample:
    outcome_hash: str
    physical_policy_hash: str
    context_hash: str
    trajectory_digest: str
    requested_aim_m: tuple[float, float, float]
    observed_delivery_m: tuple[float, float, float]
    contact_observed: bool
    safe: bool
    exact_replay: bool
    schema_version: str = "rosclaw_soccer.pass_aim_calibration_sample.v1"

    def __post_init__(self) -> None:
        values = (*self.requested_aim_m, *self.observed_delivery_m)
        if (
            not _HASH.fullmatch(self.outcome_hash)
            or not _HASH.fullmatch(self.physical_policy_hash)
            or not _HASH.fullmatch(self.context_hash)
            or not _HASH.fullmatch(self.trajectory_digest)
            or len(self.requested_aim_m) != 3
            or len(self.observed_delivery_m) != 3
            or any(not math.isfinite(value) for value in values)
            or not all(
                isinstance(value, bool)
                for value in (self.contact_observed, self.safe, self.exact_replay)
            )
        ):
            raise ValueError("pass calibration sample is invalid")

    @property
    def eligible(self) -> bool:
        return self.contact_observed and self.safe and self.exact_replay

    @property
    def residual_m(self) -> tuple[float, float, float]:
        residual = tuple(
            float(aim - delivered)
            for aim, delivered in zip(self.requested_aim_m, self.observed_delivery_m, strict=True)
        )
        return (residual[0], residual[1], residual[2])

    @property
    def sample_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["eligible"] = self.eligible
        value["residual_m"] = list(self.residual_m)
        return value


@dataclass(frozen=True)
class PassAimResidualActor:
    physical_policy_hash: str
    dataset_manifest_hash: str
    source_sample_hashes: tuple[str, ...]
    source_context_hashes: tuple[str, ...]
    source_trajectory_digests: tuple[str, ...]
    residual_mean_m: tuple[float, float, float]
    residual_rms_m: float
    maximum_compensation_m: float = 2.0
    minimum_deployment_samples: int = 4
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    schema_version: str = "rosclaw_soccer.pass_aim_residual_actor.v1"

    def __post_init__(self) -> None:
        hashes = (self.physical_policy_hash, self.dataset_manifest_hash)
        residual = np.asarray(self.residual_mean_m, dtype=np.float64)
        if (
            any(not _HASH.fullmatch(value) for value in hashes)
            or not self.source_sample_hashes
            or any(not _HASH.fullmatch(value) for value in self.source_sample_hashes)
            or len(self.source_context_hashes) != len(self.source_sample_hashes)
            or len(self.source_trajectory_digests) != len(self.source_sample_hashes)
            or any(not _HASH.fullmatch(value) for value in self.source_context_hashes)
            or any(not _HASH.fullmatch(value) for value in self.source_trajectory_digests)
            or len(set(self.source_sample_hashes)) != len(self.source_sample_hashes)
            or residual.shape != (3,)
            or not np.all(np.isfinite(residual))
            or not math.isfinite(self.residual_rms_m)
            or self.residual_rms_m < 0.0
            or not math.isfinite(self.maximum_compensation_m)
            or not 0.10 <= self.maximum_compensation_m <= 3.0
            or float(np.linalg.norm(residual)) > self.maximum_compensation_m
            or not 2 <= self.minimum_deployment_samples <= 100
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.direct_joint_torque_output
        ):
            raise ValueError("pass aim residual actor is invalid")

    @property
    def deployment_ready(self) -> bool:
        return bool(
            len(set(self.source_context_hashes)) >= self.minimum_deployment_samples
            and len(set(self.source_trajectory_digests)) >= self.minimum_deployment_samples
        )

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def propose_aim(
        self, desired_delivery_m: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        desired = np.asarray(desired_delivery_m, dtype=np.float64)
        if desired.shape != (3,) or not np.all(np.isfinite(desired)):
            raise ValueError("desired pass delivery must be a finite xyz vector")
        aim = desired + np.asarray(self.residual_mean_m, dtype=np.float64)
        return (float(aim[0]), float(aim[1]), float(aim[2]))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_sample_hashes"] = list(self.source_sample_hashes)
        value["source_context_hashes"] = list(self.source_context_hashes)
        value["source_trajectory_digests"] = list(self.source_trajectory_digests)
        value["residual_mean_m"] = list(self.residual_mean_m)
        value["deployment_ready"] = self.deployment_ready
        return value


def train_pass_aim_residual_actor(
    samples: tuple[PassAimCalibrationSample, ...],
    *,
    maximum_compensation_m: float = 2.0,
    minimum_deployment_samples: int = 4,
) -> PassAimResidualActor:
    """Fit a robust constant residual from eligible, replay-stable physics."""

    eligible = tuple(sample for sample in samples if sample.eligible)
    if not eligible:
        raise ValueError("pass calibration requires eligible physical samples")
    policy_hashes = {sample.physical_policy_hash for sample in eligible}
    if len(policy_hashes) != 1:
        raise ValueError("pass calibration cannot mix physical policy identities")
    residuals = np.asarray([sample.residual_m for sample in eligible], dtype=np.float64)
    # The median is deliberately used for the small online replay batches: a
    # single collision or late rolling sample cannot drag the correction.
    mean = np.median(residuals, axis=0)
    rms = float(np.sqrt(np.mean(np.square(residuals - mean))))
    manifest = str(
        hash_json(
            {
                "schema": PassAimCalibrationSample.schema_version,
                "samples": [sample.sample_hash for sample in eligible],
            }
        )
    )
    return PassAimResidualActor(
        physical_policy_hash=next(iter(policy_hashes)),
        dataset_manifest_hash=manifest,
        source_sample_hashes=tuple(sample.sample_hash for sample in eligible),
        source_context_hashes=tuple(sample.context_hash for sample in eligible),
        source_trajectory_digests=tuple(sample.trajectory_digest for sample in eligible),
        residual_mean_m=(float(mean[0]), float(mean[1]), float(mean[2])),
        residual_rms_m=rms,
        maximum_compensation_m=maximum_compensation_m,
        minimum_deployment_samples=minimum_deployment_samples,
    )


__all__ = [
    "PassAimCalibrationSample",
    "PassAimResidualActor",
    "train_pass_aim_residual_actor",
]
