"""Role-local calibration of a frozen finisher's high-level aim target.

The calibrated value is an intent consumed by an already qualified whole-body
kick prior.  It is not a pose, joint, torque, ROS, DDS, or hardware command.
Only precise, safe and exactly replayed CPU MuJoCo samples may enter the actor.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.runtime_finish_plan_actor import (
    PREPARED_FINISH_PLAN_FEATURE_NAMES,
)
from rosclaw_soccer.sim.contracts import hash_json

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_FEATURE_COUNT = len(PREPARED_FINISH_PLAN_FEATURE_NAMES)
_YAW_INDEX = PREPARED_FINISH_PLAN_FEATURE_NAMES.index("passer_yaw_rad")
_SCALE_FLOOR = np.asarray(
    (0.02, 0.02, 0.002, 0.002, 0.002, 0.02, 0.005, 0.005, 0.01),
    dtype=np.float64,
)


def _commitment(value: str, label: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{label} must be a SHA-256 commitment")
    return value


def _finite_xyz(value: tuple[float, ...], label: str) -> NDArray[np.float64]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must be finite xyz")
    return vector


@dataclass(frozen=True)
class FinishTargetCalibrationSample:
    """One content-bound target calibration outcome from CPU MuJoCo."""

    context_hash: str
    trajectory_hash: str
    control_envelope_hash: str
    features: tuple[float, ...]
    requested_physical_target_m: tuple[float, float, float]
    executed_policy_target_m: tuple[float, float, float]
    executed_foot_yaw_offset_rad: float
    observed_crossing_m: tuple[float, float, float]
    target_error_m: float
    safe: bool
    exact_replay: bool
    source_partition: str = "DEVELOPMENT"
    physics_authority: str = "CPU_MUJOCO"
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False

    def __post_init__(self) -> None:
        _commitment(self.context_hash, "context_hash")
        _commitment(self.trajectory_hash, "trajectory_hash")
        _commitment(self.control_envelope_hash, "control_envelope_hash")
        features = np.asarray(self.features, dtype=np.float64)
        requested = _finite_xyz(self.requested_physical_target_m, "requested target")
        policy = _finite_xyz(self.executed_policy_target_m, "policy target")
        observed = _finite_xyz(self.observed_crossing_m, "observed crossing")
        measured_error = float(np.linalg.norm(observed - requested))
        if (
            features.shape != (_FEATURE_COUNT,)
            or not np.all(np.isfinite(features))
            or not math.isfinite(self.target_error_m)
            or self.target_error_m < 0.0
            or not math.isfinite(self.executed_foot_yaw_offset_rad)
            or abs(self.executed_foot_yaw_offset_rad) > 0.12
            or abs(measured_error - self.target_error_m) > 1.0e-6
            or not 5.0 <= requested[0] <= 15.0
            or not 5.0 <= policy[0] <= 15.0
            or abs(requested[1]) > 5.0
            or abs(policy[1]) > 5.0
            or not 0.05 <= requested[2] <= 2.50
            or not 0.05 <= policy[2] <= 2.50
            or self.source_partition != "DEVELOPMENT"
            or self.physics_authority != "CPU_MUJOCO"
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_command_sent
        ):
            raise ValueError("finish target calibration sample is invalid")

    @property
    def precise(self) -> bool:
        return bool(self.safe and self.exact_replay and self.target_error_m <= 0.10)

    @property
    def residual_m(self) -> tuple[float, float, float]:
        requested = np.asarray(self.requested_physical_target_m, dtype=np.float64)
        policy = np.asarray(self.executed_policy_target_m, dtype=np.float64)
        residual = policy - requested
        return (float(residual[0]), float(residual[1]), float(residual[2]))

    @property
    def sample_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class FinishTargetFailureMemory:
    """One bounded-search failure basin that the actor must not interpolate across."""

    context_hash: str
    search_hash: str
    control_envelope_hash: str
    features: tuple[float, ...]
    failure_code: str
    candidate_count: int
    safe_candidate_count: int
    best_safe_target_error_m: float | None
    exact_replay: bool
    source_partition: str = "DEVELOPMENT_DISCOVERY"
    physics_authority: str = "CPU_MUJOCO"
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False

    def __post_init__(self) -> None:
        _commitment(self.context_hash, "context_hash")
        _commitment(self.search_hash, "search_hash")
        _commitment(self.control_envelope_hash, "control_envelope_hash")
        features = np.asarray(self.features, dtype=np.float64)
        error = self.best_safe_target_error_m
        if (
            features.shape != (_FEATURE_COUNT,)
            or not np.all(np.isfinite(features))
            or self.failure_code != "NO_PRECISE_SAFE_ACTION_IN_BOUNDED_SEARCH"
            or isinstance(self.candidate_count, bool)
            or isinstance(self.safe_candidate_count, bool)
            or not 4 <= self.candidate_count <= 64
            or not 0 <= self.safe_candidate_count <= self.candidate_count
            or (error is not None and (not math.isfinite(error) or error <= 0.10))
            or not self.exact_replay
            or self.source_partition != "DEVELOPMENT_DISCOVERY"
            or self.physics_authority != "CPU_MUJOCO"
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_command_sent
        ):
            raise ValueError("finish target failure memory is invalid")

    @property
    def failure_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class ContextualFinishTargetDecision:
    accepted: bool
    route: str
    policy_target_m: tuple[float, float, float] | None
    foot_yaw_offset_rad: float | None
    nearest_support_distance: float | None
    nearest_failure_distance: float | None
    supporting_context_hashes: tuple[str, ...]
    actor_hash: str
    activation_ceiling: str = "SIM_ONLY"
    direct_joint_torque_output: bool = False


@dataclass(frozen=True)
class G1ContextualFinishTargetActor:
    """Data-driven target residual with explicit evidence and OOD gates."""

    body_hash: str
    kick_prior_hash: str
    roster_hash: str
    finisher_self_model_hash: str
    control_envelope_hash: str
    source_evidence_hashes: tuple[str, ...]
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    samples: tuple[FinishTargetCalibrationSample, ...]
    failure_memories: tuple[FinishTargetFailureMemory, ...] = ()
    nearest_sample_count: int = 3
    maximum_support_distance: float = 3.0
    failure_exclusion_distance: float = 0.40
    minimum_distinct_contexts: int = 4
    minimum_distinct_trajectories: int = 4
    agent_id: str = "red.finisher"
    owned_skill: str = "contextual_finish_target"
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    online_hot_swap_allowed: bool = False
    schema_version: str = "rosclaw_soccer.contextual_finish_target_actor.v2"

    def __post_init__(self) -> None:
        for value, label in (
            (self.body_hash, "body_hash"),
            (self.kick_prior_hash, "kick_prior_hash"),
            (self.roster_hash, "roster_hash"),
            (self.finisher_self_model_hash, "finisher_self_model_hash"),
            (self.control_envelope_hash, "control_envelope_hash"),
        ):
            _commitment(value, label)
        for value in self.source_evidence_hashes:
            _commitment(value, "source_evidence_hash")
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        context_count = len({sample.context_hash for sample in self.samples})
        trajectory_count = len({sample.trajectory_hash for sample in self.samples})
        if (
            not self.source_evidence_hashes
            or center.shape != (_FEATURE_COUNT,)
            or scale.shape != (_FEATURE_COUNT,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
            or not self.samples
            or any(not sample.precise for sample in self.samples)
            or any(
                sample.control_envelope_hash != self.control_envelope_hash
                for sample in self.samples
            )
            or any(
                memory.control_envelope_hash != self.control_envelope_hash
                for memory in self.failure_memories
            )
            or len({memory.context_hash for memory in self.failure_memories})
            != len(self.failure_memories)
            or {memory.context_hash for memory in self.failure_memories}.intersection(
                sample.context_hash for sample in self.samples
            )
            or context_count != len(self.samples)
            or trajectory_count != len(self.samples)
            or not 1 <= self.nearest_sample_count <= 8
            or not 0.25 <= self.maximum_support_distance <= 4.0
            or not 0.10 <= self.failure_exclusion_distance <= 2.0
            or not 4 <= self.minimum_distinct_contexts <= 32
            or not 4 <= self.minimum_distinct_trajectories <= 32
            or self.schema_version != "rosclaw_soccer.contextual_finish_target_actor.v2"
            or (self.agent_id, self.owned_skill) != ("red.finisher", "contextual_finish_target")
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.direct_joint_torque_output
            or self.online_hot_swap_allowed
        ):
            raise ValueError("contextual finish target actor violates its contract")

    @property
    def distinct_context_count(self) -> int:
        return len({sample.context_hash for sample in self.samples})

    @property
    def distinct_trajectory_count(self) -> int:
        return len({sample.trajectory_hash for sample in self.samples})

    @property
    def evidence_ready(self) -> bool:
        return bool(
            self.distinct_context_count >= self.minimum_distinct_contexts
            and self.distinct_trajectory_count >= self.minimum_distinct_trajectories
        )

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            **asdict(self),
            "feature_names": list(PREPARED_FINISH_PLAN_FEATURE_NAMES),
            "algorithm": (
                "nearest_verified_finish_expert_with_failure_veto"
                if self.nearest_sample_count == 1
                else "local_weighted_median_finish_intent_residual_with_failure_veto"
            ),
            "authority": "HIGH_LEVEL_FINISH_TARGET_AND_CONTACT_NORMAL_ONLY",
            "evidence_ready": self.evidence_ready,
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(
        self,
        features: tuple[float, ...],
        requested_physical_target_m: tuple[float, float, float],
    ) -> ContextualFinishTargetDecision:
        vector = np.asarray(features, dtype=np.float64)
        requested = _finite_xyz(requested_physical_target_m, "requested target")
        if vector.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(vector)):
            raise ValueError("contextual finish target features are invalid")
        if not self.evidence_ready:
            return ContextualFinishTargetDecision(
                accepted=False,
                route="INSUFFICIENT_DISTINCT_PHYSICAL_SUPPORT",
                policy_target_m=None,
                foot_yaw_offset_rad=None,
                nearest_support_distance=None,
                nearest_failure_distance=None,
                supporting_context_hashes=(),
                actor_hash=self.actor_hash,
            )
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        failure_ranked = sorted(
            [
                (
                    self._feature_distance(vector, memory.features, scale),
                    memory,
                )
                for memory in self.failure_memories
            ],
            key=lambda item: (item[0], item[1].failure_hash),
        )
        nearest_failure_distance = failure_ranked[0][0] if failure_ranked else None
        if (
            nearest_failure_distance is not None
            and nearest_failure_distance <= self.failure_exclusion_distance
        ):
            return ContextualFinishTargetDecision(
                accepted=False,
                route="KNOWN_FINISH_FAILURE_BASIN_FALLBACK",
                policy_target_m=None,
                foot_yaw_offset_rad=None,
                nearest_support_distance=None,
                nearest_failure_distance=nearest_failure_distance,
                supporting_context_hashes=(failure_ranked[0][1].context_hash,),
                actor_hash=self.actor_hash,
            )
        ranked: list[tuple[float, FinishTargetCalibrationSample]] = []
        for sample in self.samples:
            ranked.append((self._feature_distance(vector, sample.features, scale), sample))
        ranked.sort(key=lambda item: (item[0], item[1].sample_hash))
        nearest_distance = ranked[0][0]
        selected = ranked[: min(self.nearest_sample_count, len(ranked))]
        if nearest_distance > self.maximum_support_distance:
            return ContextualFinishTargetDecision(
                accepted=False,
                route="CONTEXTUAL_FINISH_TARGET_OOD_FALLBACK",
                policy_target_m=None,
                foot_yaw_offset_rad=None,
                nearest_support_distance=nearest_distance,
                nearest_failure_distance=nearest_failure_distance,
                supporting_context_hashes=(),
                actor_hash=self.actor_hash,
            )
        weights = np.asarray([math.exp(-0.5 * distance**2) for distance, _ in selected])
        residuals = np.asarray([sample.residual_m for _, sample in selected])
        # Coordinate-wise weighted median is robust to one locally bad calibration.
        residual = np.asarray(
            [_weighted_median(residuals[:, index], weights) for index in range(3)]
        )
        foot_yaw = _weighted_median(
            np.asarray(
                [sample.executed_foot_yaw_offset_rad for _, sample in selected],
                dtype=np.float64,
            ),
            weights,
        )
        policy_target = requested + residual
        if (
            not 5.0 <= policy_target[0] <= 15.0
            or abs(policy_target[1]) > 5.0
            or not 0.05 <= policy_target[2] <= 2.50
        ):
            return ContextualFinishTargetDecision(
                accepted=False,
                route="CONTEXTUAL_FINISH_TARGET_ENVELOPE_FALLBACK",
                policy_target_m=None,
                foot_yaw_offset_rad=None,
                nearest_support_distance=nearest_distance,
                nearest_failure_distance=nearest_failure_distance,
                supporting_context_hashes=tuple(sample.context_hash for _, sample in selected),
                actor_hash=self.actor_hash,
            )
        return ContextualFinishTargetDecision(
            accepted=True,
            route="VERIFIED_CONTEXTUAL_FINISH_TARGET",
            policy_target_m=(
                float(policy_target[0]),
                float(policy_target[1]),
                float(policy_target[2]),
            ),
            foot_yaw_offset_rad=foot_yaw,
            nearest_support_distance=nearest_distance,
            nearest_failure_distance=nearest_failure_distance,
            supporting_context_hashes=tuple(sample.context_hash for _, sample in selected),
            actor_hash=self.actor_hash,
        )

    @staticmethod
    def _feature_distance(query: np.ndarray, sample: tuple[float, ...], scale: np.ndarray) -> float:
        delta = np.asarray(sample, dtype=np.float64) - query
        delta[_YAW_INDEX] = math.atan2(math.sin(delta[_YAW_INDEX]), math.cos(delta[_YAW_INDEX]))
        return float(np.linalg.norm(delta / scale))


def fit_contextual_finish_target_actor(
    *,
    body_hash: str,
    kick_prior_hash: str,
    roster_hash: str,
    finisher_self_model_hash: str,
    control_envelope_hash: str,
    source_evidence_hashes: tuple[str, ...],
    samples: tuple[FinishTargetCalibrationSample, ...],
    failure_memories: tuple[FinishTargetFailureMemory, ...] = (),
) -> G1ContextualFinishTargetActor:
    """Fit a bounded actor; a small seed remains non-deployable by design."""

    if not samples:
        raise ValueError("contextual finish target fitting needs physical samples")
    matrix = np.asarray([sample.features for sample in samples], dtype=np.float64)
    yaw = matrix[:, _YAW_INDEX]
    center = np.mean(matrix, axis=0)
    center[_YAW_INDEX] = math.atan2(float(np.mean(np.sin(yaw))), float(np.mean(np.cos(yaw))))
    unwrapped = matrix.copy()
    unwrapped[:, _YAW_INDEX] = center[_YAW_INDEX] + np.arctan2(
        np.sin(yaw - center[_YAW_INDEX]), np.cos(yaw - center[_YAW_INDEX])
    )
    scale = np.maximum(np.std(unwrapped, axis=0), _SCALE_FLOOR)
    return G1ContextualFinishTargetActor(
        body_hash=body_hash,
        kick_prior_hash=kick_prior_hash,
        roster_hash=roster_hash,
        finisher_self_model_hash=finisher_self_model_hash,
        control_envelope_hash=control_envelope_hash,
        source_evidence_hashes=source_evidence_hashes,
        feature_center=tuple(float(value) for value in center),
        feature_scale=tuple(float(value) for value in scale),
        samples=samples,
        failure_memories=failure_memories,
    )


def save_contextual_finish_target_actor(actor: G1ContextualFinishTargetActor, path: Path) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def load_contextual_finish_target_actor(path: Path) -> G1ContextualFinishTargetActor:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("contextual finish target actor artifact must be an object")
    claimed_hash = value.pop("actor_hash", None)
    value.pop("feature_names", None)
    value.pop("algorithm", None)
    value.pop("authority", None)
    value.pop("evidence_ready", None)
    raw_samples = value.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("contextual finish target actor samples are invalid")
    value["source_evidence_hashes"] = tuple(value["source_evidence_hashes"])
    value["feature_center"] = tuple(value["feature_center"])
    value["feature_scale"] = tuple(value["feature_scale"])
    value["samples"] = tuple(
        FinishTargetCalibrationSample(
            **{
                **sample,
                "features": tuple(sample["features"]),
                "requested_physical_target_m": tuple(sample["requested_physical_target_m"]),
                "executed_policy_target_m": tuple(sample["executed_policy_target_m"]),
                "observed_crossing_m": tuple(sample["observed_crossing_m"]),
            }
        )
        for sample in raw_samples
        if isinstance(sample, dict)
    )
    if len(value["samples"]) != len(raw_samples):
        raise ValueError("contextual finish target actor sample entry is invalid")
    raw_failures = value.get("failure_memories", [])
    if not isinstance(raw_failures, list):
        raise ValueError("contextual finish target actor failure memories are invalid")
    value["failure_memories"] = tuple(
        FinishTargetFailureMemory(
            **{
                **memory,
                "features": tuple(memory["features"]),
            }
        )
        for memory in raw_failures
        if isinstance(memory, dict)
    )
    if len(value["failure_memories"]) != len(raw_failures):
        raise ValueError("contextual finish target actor failure memory entry is invalid")
    actor = G1ContextualFinishTargetActor(**value)
    if claimed_hash != actor.actor_hash:
        raise ValueError("contextual finish target actor integrity mismatch")
    return actor


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = 0.5 * float(np.sum(sorted_weights))
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


__all__ = [
    "ContextualFinishTargetDecision",
    "FinishTargetCalibrationSample",
    "FinishTargetFailureMemory",
    "G1ContextualFinishTargetActor",
    "fit_contextual_finish_target_actor",
    "load_contextual_finish_target_actor",
    "save_contextual_finish_target_actor",
]
