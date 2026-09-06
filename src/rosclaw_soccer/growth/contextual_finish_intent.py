"""Evidence-bound local experts for stable high-level finish intent.

The actor selects only an exactly replayed, safe high-level action inside a
small measured context neighbourhood.  It deliberately does not interpolate:
contact timing is discontinuous and a smooth parameter blend can cross an
unsafe or inaccurate basin.  Joint and torque authority stays with the frozen
whole-body prior.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.runtime_finish_plan_actor import (
    PREPARED_FINISH_PLAN_FEATURE_NAMES,
)
from rosclaw_soccer.sim.contracts import hash_json

CONTEXTUAL_FINISH_INTENT_FEATURE_NAMES = (
    "receiver_phase_start_sec",
    *PREPARED_FINISH_PLAN_FEATURE_NAMES,
)
DEFAULT_CONTEXTUAL_FINISH_INTENT_SCALE = (
    0.0005,
    0.00025,
    0.00075,
    0.02,
    0.02,
    0.01,
    0.00025,
    0.005,
    0.005,
    0.01,
)
_FEATURE_COUNT = len(CONTEXTUAL_FINISH_INTENT_FEATURE_NAMES)
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _commitment(value: str, label: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{label} must be a SHA-256 commitment")
    return value


def contextual_finish_intent_features(
    *,
    receiver_phase_start_sec: float,
    prepared_features: Sequence[float],
) -> tuple[float, ...]:
    """Bind contact phase to the existing pre-rollout team observation."""

    vector = np.asarray(prepared_features, dtype=np.float64)
    if (
        not math.isfinite(receiver_phase_start_sec)
        or vector.shape != (len(PREPARED_FINISH_PLAN_FEATURE_NAMES),)
        or not np.all(np.isfinite(vector))
    ):
        raise ValueError("contextual finish intent features must be finite")
    return (float(receiver_phase_start_sec), *(float(value) for value in vector))


@dataclass(frozen=True)
class ContextualFinishIntentAction:
    """Four interpretable parameters consumed by a frozen kick prior."""

    policy_target_y_m: float
    foot_yaw_offset_rad: float
    stance_offset_y_m: float
    foot_pitch_offset_rad: float

    def __post_init__(self) -> None:
        values = (
            self.policy_target_y_m,
            self.foot_yaw_offset_rad,
            self.stance_offset_y_m,
            self.foot_pitch_offset_rad,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or not -1.0 <= self.policy_target_y_m <= 1.0
            or not -0.12 <= self.foot_yaw_offset_rad <= 0.12
            or not -0.12 <= self.stance_offset_y_m <= 0.12
            or not -0.10 <= self.foot_pitch_offset_rad <= 0.15
        ):
            raise ValueError("contextual finish intent action is invalid")

    @property
    def values(self) -> tuple[float, float, float, float]:
        return (
            self.policy_target_y_m,
            self.foot_yaw_offset_rad,
            self.stance_offset_y_m,
            self.foot_pitch_offset_rad,
        )

    @property
    def action_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class ContextualFinishIntentSample:
    """One successful CPU MuJoCo expert with immutable evidence lineage."""

    context_hash: str
    trajectory_hash: str
    control_envelope_hash: str
    source_evidence_hash: str
    features: tuple[float, ...]
    action: ContextualFinishIntentAction
    requested_physical_target_m: tuple[float, float, float]
    observed_crossing_m: tuple[float, float, float]
    target_error_m: float
    safe: bool
    stability_retained: bool
    exact_replay: bool
    source_partition: str = "DEVELOPMENT"
    physics_authority: str = "CPU_MUJOCO"
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False

    def __post_init__(self) -> None:
        for value, label in (
            (self.context_hash, "context_hash"),
            (self.trajectory_hash, "trajectory_hash"),
            (self.control_envelope_hash, "control_envelope_hash"),
            (self.source_evidence_hash, "source_evidence_hash"),
        ):
            _commitment(value, label)
        features = np.asarray(self.features, dtype=np.float64)
        requested = np.asarray(self.requested_physical_target_m, dtype=np.float64)
        observed = np.asarray(self.observed_crossing_m, dtype=np.float64)
        measured_error = float(np.linalg.norm(requested[1:] - observed[1:]))
        if (
            features.shape != (_FEATURE_COUNT,)
            or not np.all(np.isfinite(features))
            or requested.shape != (3,)
            or observed.shape != (3,)
            or not np.all(np.isfinite(requested))
            or not np.all(np.isfinite(observed))
            or not math.isfinite(self.target_error_m)
            or self.target_error_m < 0.0
            or abs(measured_error - self.target_error_m) > 1.0e-6
            or self.target_error_m > 0.10
            or not self.safe
            or not self.stability_retained
            or not self.exact_replay
            or self.source_partition != "DEVELOPMENT"
            or self.physics_authority != "CPU_MUJOCO"
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_command_sent
        ):
            raise ValueError("contextual finish intent sample is invalid")

    @property
    def sample_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class ContextualFinishIntentDecision:
    accepted: bool
    route: str
    action: ContextualFinishIntentAction | None
    nearest_support_distance: float | None
    supporting_context_hash: str | None
    actor_hash: str
    activation_ceiling: str = "SIM_ONLY"
    direct_joint_torque_output: bool = False

    def __post_init__(self) -> None:
        if (
            self.accepted != (self.action is not None)
            or self.accepted != (self.supporting_context_hash is not None)
            or self.route
            not in {
                "INSUFFICIENT_DISTINCT_PHYSICAL_SUPPORT",
                "CONTEXTUAL_FINISH_INTENT_OOD_FALLBACK",
                "VERIFIED_LOCAL_FINISH_INTENT_EXPERT",
            }
            or (self.accepted and self.route != "VERIFIED_LOCAL_FINISH_INTENT_EXPERT")
            or (
                self.nearest_support_distance is not None
                and (
                    not math.isfinite(self.nearest_support_distance)
                    or self.nearest_support_distance < 0.0
                )
            )
            or (
                self.supporting_context_hash is not None
                and not _HASH.fullmatch(self.supporting_context_hash)
            )
            or not _HASH.fullmatch(self.actor_hash)
            or self.activation_ceiling != "SIM_ONLY"
            or self.direct_joint_torque_output
        ):
            raise ValueError("contextual finish intent decision is invalid")


@dataclass(frozen=True)
class ContextualFinishIntentActor:
    """Nearest verified 4-D expert with a tight fail-closed support gate."""

    body_hash: str
    kick_prior_hash: str
    roster_hash: str
    finisher_self_model_hash: str
    control_envelope_hash: str
    source_evidence_hashes: tuple[str, ...]
    feature_scale: tuple[float, ...]
    samples: tuple[ContextualFinishIntentSample, ...]
    maximum_support_distance: float = 0.35
    minimum_distinct_contexts: int = 4
    minimum_distinct_trajectories: int = 4
    agent_id: str = "red.finisher"
    owned_skill: str = "contextual_finish_intent"
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    online_hot_swap_allowed: bool = False
    schema_version: str = "rosclaw_soccer.contextual_finish_intent_actor.v1"

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
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        contexts = {sample.context_hash for sample in self.samples}
        trajectories = {sample.trajectory_hash for sample in self.samples}
        if (
            not self.source_evidence_hashes
            or scale.shape != (_FEATURE_COUNT,)
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
            or not self.samples
            or len(contexts) != len(self.samples)
            or len(trajectories) != len(self.samples)
            or any(
                sample.control_envelope_hash != self.control_envelope_hash
                for sample in self.samples
            )
            or any(
                sample.source_evidence_hash not in self.source_evidence_hashes
                for sample in self.samples
            )
            or not np.array_equal(scale, np.asarray(DEFAULT_CONTEXTUAL_FINISH_INTENT_SCALE))
            or not 0.05 <= self.maximum_support_distance <= 0.35
            or not 4 <= self.minimum_distinct_contexts <= 64
            or not 4 <= self.minimum_distinct_trajectories <= 64
            or (self.agent_id, self.owned_skill) != ("red.finisher", "contextual_finish_intent")
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.direct_joint_torque_output
            or self.online_hot_swap_allowed
            or self.schema_version != "rosclaw_soccer.contextual_finish_intent_actor.v1"
        ):
            raise ValueError("contextual finish intent actor violates its contract")

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
            "feature_names": list(CONTEXTUAL_FINISH_INTENT_FEATURE_NAMES),
            "algorithm": "nearest_verified_4d_expert_no_interpolation",
            "authority": "HIGH_LEVEL_FINISH_INTENT_ONLY",
            "evidence_ready": self.evidence_ready,
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(self, features: tuple[float, ...]) -> ContextualFinishIntentDecision:
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(vector)):
            raise ValueError("contextual finish intent query is invalid")
        if not self.evidence_ready:
            return self._rejection("INSUFFICIENT_DISTINCT_PHYSICAL_SUPPORT")
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        ranked = sorted(
            (
                (float(np.linalg.norm((vector - np.asarray(sample.features)) / scale)), sample)
                for sample in self.samples
            ),
            key=lambda item: (item[0], item[1].sample_hash),
        )
        distance, sample = ranked[0]
        if distance > self.maximum_support_distance:
            return ContextualFinishIntentDecision(
                accepted=False,
                route="CONTEXTUAL_FINISH_INTENT_OOD_FALLBACK",
                action=None,
                nearest_support_distance=distance,
                supporting_context_hash=None,
                actor_hash=self.actor_hash,
            )
        return ContextualFinishIntentDecision(
            accepted=True,
            route="VERIFIED_LOCAL_FINISH_INTENT_EXPERT",
            action=sample.action,
            nearest_support_distance=distance,
            supporting_context_hash=sample.context_hash,
            actor_hash=self.actor_hash,
        )

    def _rejection(self, route: str) -> ContextualFinishIntentDecision:
        return ContextualFinishIntentDecision(
            accepted=False,
            route=route,
            action=None,
            nearest_support_distance=None,
            supporting_context_hash=None,
            actor_hash=self.actor_hash,
        )


def save_contextual_finish_intent_actor(actor: ContextualFinishIntentActor, path: Path) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def load_contextual_finish_intent_actor(path: Path) -> ContextualFinishIntentActor:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("contextual finish intent actor artifact must be an object")
    claimed_hash = value.pop("actor_hash", None)
    for key in ("feature_names", "algorithm", "authority", "evidence_ready"):
        value.pop(key, None)
    raw_samples = value.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("contextual finish intent actor samples are invalid")
    samples: list[ContextualFinishIntentSample] = []
    for raw in raw_samples:
        if not isinstance(raw, dict) or not isinstance(raw.get("action"), dict):
            raise ValueError("contextual finish intent actor sample entry is invalid")
        samples.append(
            ContextualFinishIntentSample(
                **{
                    **raw,
                    "features": tuple(raw["features"]),
                    "action": ContextualFinishIntentAction(**raw["action"]),
                    "requested_physical_target_m": tuple(raw["requested_physical_target_m"]),
                    "observed_crossing_m": tuple(raw["observed_crossing_m"]),
                }
            )
        )
    value["source_evidence_hashes"] = tuple(value["source_evidence_hashes"])
    value["feature_scale"] = tuple(value["feature_scale"])
    value["samples"] = tuple(samples)
    actor = ContextualFinishIntentActor(**value)
    if claimed_hash != actor.actor_hash:
        raise ValueError("contextual finish intent actor integrity mismatch")
    return actor


__all__ = [
    "CONTEXTUAL_FINISH_INTENT_FEATURE_NAMES",
    "DEFAULT_CONTEXTUAL_FINISH_INTENT_SCALE",
    "ContextualFinishIntentAction",
    "ContextualFinishIntentActor",
    "ContextualFinishIntentDecision",
    "ContextualFinishIntentSample",
    "contextual_finish_intent_features",
    "load_contextual_finish_intent_actor",
    "save_contextual_finish_intent_actor",
]
