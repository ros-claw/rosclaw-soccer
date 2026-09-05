"""Risk-sensitive ensemble actor for causal successor-skill entry.

Unlike the first regression actor, this actor keeps the discrete phases that
were actually probed in physics.  Four independently trained heads estimate
both body safety and complete-chain success for every candidate.  A learned
phase receives authority only when every head clears the safety threshold and
its conservative success estimate exceeds the frozen parent.  Otherwise the
actor returns the parent phase.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.causal_skill_transition import (
    CAUSAL_TRANSITION_FEATURE_NAMES,
    CausalTransitionDecision,
)
from rosclaw_soccer.sim.contracts import hash_json

RISK_CANDIDATE_POLICY_FRAMES = (84, 86, 88, 90, 93)
_FEATURE_COUNT = len(CAUSAL_TRANSITION_FEATURE_NAMES)
_RISK_INPUT_COUNT = _FEATURE_COUNT + 1


@dataclass(frozen=True)
class CausalTransitionProbeSample:
    """One context/phase counterfactual measured in CPU MuJoCo."""

    sample_id: str
    context_id: str
    features: tuple[float, ...]
    trigger_policy_frame: int
    safe: bool
    chain_passed: bool
    source_report_hash: str
    source_probe_hash: str
    schema_version: str = "rosclaw.growth.causal_transition_probe_sample.v1"

    def __post_init__(self) -> None:
        values = np.asarray(self.features, dtype=np.float64)
        commitments = (self.source_report_hash, self.source_probe_hash)
        if (
            not self.sample_id
            or not isinstance(self.sample_id, str)
            or not self.context_id
            or not isinstance(self.context_id, str)
            or values.shape != (_FEATURE_COUNT,)
            or not np.all(np.isfinite(values))
            or self.trigger_policy_frame not in RISK_CANDIDATE_POLICY_FRAMES
            or self.chain_passed
            and not self.safe
            or type(self.safe) is not bool
            or type(self.chain_passed) is not bool
            or any(
                not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71
                for value in commitments
            )
            or self.schema_version != "rosclaw.growth.causal_transition_probe_sample.v1"
        ):
            raise ValueError("causal transition probe sample is malformed")


@dataclass(frozen=True)
class G1CausalSkillTransitionRiskActor:
    """Four-head, fail-closed phase selector with no joint authority."""

    source_snapshot_hash: str
    training_snapshot_hash: str
    implementation_hash: str
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    feature_minimum: tuple[float, ...]
    feature_maximum: tuple[float, ...]
    hidden_weights: tuple[tuple[tuple[float, ...], ...], ...]
    hidden_bias: tuple[tuple[float, ...], ...]
    output_weights: tuple[tuple[tuple[float, ...], ...], ...]
    output_bias: tuple[tuple[float, ...], ...]
    head_training_context_ids: tuple[tuple[str, ...], ...]
    head_validation_context_ids: tuple[tuple[str, ...], ...]
    safety_probability_threshold: float
    chain_probability_threshold: float
    minimum_chain_advantage: float
    candidate_policy_frames: tuple[int, ...] = RISK_CANDIDATE_POLICY_FRAMES
    parent_trigger_policy_frame: int = 88
    candidate_frame_scale: float = 5.0
    maximum_ood_distance: float = 2.0
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    schema_version: str = "rosclaw.growth.causal_transition_risk_actor.v2"

    def __post_init__(self) -> None:
        commitments = (
            self.source_snapshot_hash,
            self.training_snapshot_hash,
            self.implementation_hash,
        )
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        minimum = np.asarray(self.feature_minimum, dtype=np.float64)
        maximum = np.asarray(self.feature_maximum, dtype=np.float64)
        hidden_weights = np.asarray(self.hidden_weights, dtype=np.float64)
        hidden_bias = np.asarray(self.hidden_bias, dtype=np.float64)
        output_weights = np.asarray(self.output_weights, dtype=np.float64)
        output_bias = np.asarray(self.output_bias, dtype=np.float64)
        head_count = hidden_weights.shape[0] if hidden_weights.ndim == 3 else 0
        hidden_size = hidden_weights.shape[1] if head_count else 0
        arrays = (
            center,
            scale,
            minimum,
            maximum,
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
        )
        context_universe: set[str] = set()
        if len(self.head_training_context_ids) == 4 and len(self.head_validation_context_ids) == 4:
            context_universe = set(self.head_training_context_ids[0]) | set(
                self.head_validation_context_ids[0]
            )
        validation_union: set[str] = set().union(*map(set, self.head_validation_context_ids))
        if any(
            not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71
            for value in commitments
        ):
            raise ValueError("causal transition risk commitments must be SHA-256 hashes")
        if (
            center.shape != (_FEATURE_COUNT,)
            or scale.shape != (_FEATURE_COUNT,)
            or minimum.shape != (_FEATURE_COUNT,)
            or maximum.shape != (_FEATURE_COUNT,)
            or head_count != 4
            or not 4 <= hidden_size <= 64
            or hidden_weights.shape != (head_count, hidden_size, _RISK_INPUT_COUNT)
            or hidden_bias.shape != (head_count, hidden_size)
            or output_weights.shape != (head_count, 2, hidden_size)
            or output_bias.shape != (head_count, 2)
            or not all(np.all(np.isfinite(value)) for value in arrays)
            or np.any(scale <= 0.0)
            or np.any(minimum > maximum)
            or len(self.head_training_context_ids) != 4
            or len(self.head_validation_context_ids) != 4
            or any(len(values) < 8 for values in self.head_training_context_ids)
            or any(len(values) < 2 for values in self.head_validation_context_ids)
            or any(
                not isinstance(context_id, str) or not context_id
                for fold in (*self.head_training_context_ids, *self.head_validation_context_ids)
                for context_id in fold
            )
            or any(
                len(set(training)) != len(training)
                or len(set(validation)) != len(validation)
                or set(training) & set(validation)
                for training, validation in zip(
                    self.head_training_context_ids,
                    self.head_validation_context_ids,
                    strict=True,
                )
            )
            or any(
                set(training) | set(validation) != context_universe
                for training, validation in zip(
                    self.head_training_context_ids,
                    self.head_validation_context_ids,
                    strict=True,
                )
            )
            or sum(len(values) for values in self.head_validation_context_ids)
            != len(validation_union)
            or validation_union != context_universe
        ):
            raise ValueError("causal transition risk tensors or context split are invalid")
        if (
            tuple(self.candidate_policy_frames) != RISK_CANDIDATE_POLICY_FRAMES
            or self.parent_trigger_policy_frame not in self.candidate_policy_frames
            or not 1.0 <= self.candidate_frame_scale <= 20.0
            or not 0.5 <= self.safety_probability_threshold <= 0.999
            or not 0.2 <= self.chain_probability_threshold <= 0.999
            or not 0.0 <= self.minimum_chain_advantage <= 0.5
            or not 0.1 <= self.maximum_ood_distance <= 5.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.direct_joint_torque_output
            or self.schema_version != "rosclaw.growth.causal_transition_risk_actor.v2"
        ):
            raise ValueError("causal transition risk actor violates its safety contract")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            **asdict(self),
            "feature_names": list(CAUSAL_TRANSITION_FEATURE_NAMES),
            "algorithm": "four_head_risk_sensitive_discrete_phase_selector_v1",
            "serialized_executable_code": False,
            "pixels_used_for_training": False,
            "output_authority": "BOUNDED_SUCCESSOR_POLICY_PHASE_ONLY",
            "ood_route": "FROZEN_PARENT_TRIGGER",
            "uncertainty_rule": "MINIMUM_PROBABILITY_ACROSS_FOUR_HEADS",
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(self, features: NDArray[np.float64]) -> CausalTransitionDecision:
        observation = np.asarray(features, dtype=np.float64)
        if observation.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(observation)):
            raise ValueError("causal transition risk actor requires fourteen finite features")
        minimum = np.asarray(self.feature_minimum, dtype=np.float64)
        maximum = np.asarray(self.feature_maximum, dtype=np.float64)
        span = np.maximum(maximum - minimum, 0.05)
        support_distance = float(
            np.linalg.norm(
                (np.maximum(minimum - observation, 0.0) + np.maximum(observation - maximum, 0.0))
                / span
            )
        )
        if support_distance > self.maximum_ood_distance:
            return CausalTransitionDecision(
                accepted=False,
                trigger_policy_frame=self.parent_trigger_policy_frame,
                parent_trigger_policy_frame=self.parent_trigger_policy_frame,
                residual_frames=0,
                support_distance=support_distance,
                actor_hash=self.actor_hash,
                used_parent_fallback=True,
            )
        probabilities = self._candidate_probabilities(observation)
        return self._decision_from_probabilities(observation, probabilities, support_distance)

    def _decision_from_probabilities(
        self,
        observation: NDArray[np.float64],
        probabilities: NDArray[np.float64],
        support_distance: float,
    ) -> CausalTransitionDecision:
        if (
            observation.shape != (_FEATURE_COUNT,)
            or probabilities.ndim != 3
            or probabilities.shape[1:] != (len(self.candidate_policy_frames), 2)
            or probabilities.shape[0] < 1
        ):
            raise ValueError("causal transition risk probability tensor is invalid")
        parent_index = self.candidate_policy_frames.index(self.parent_trigger_policy_frame)
        parent_chain_floor = float(np.min(probabilities[:, parent_index, 1]))
        eligible: list[tuple[float, float, int, int]] = []
        for index, frame in enumerate(self.candidate_policy_frames):
            if frame == self.parent_trigger_policy_frame:
                continue
            safe_floor = float(np.min(probabilities[:, index, 0]))
            chain_floor = float(np.min(probabilities[:, index, 1]))
            if (
                safe_floor >= self.safety_probability_threshold
                and chain_floor >= self.chain_probability_threshold
                and chain_floor >= parent_chain_floor + self.minimum_chain_advantage
            ):
                eligible.append(
                    (chain_floor, safe_floor, -abs(frame - self.parent_trigger_policy_frame), frame)
                )
        frame = self.parent_trigger_policy_frame
        used_fallback = True
        if eligible:
            frame = max(eligible)[3]
            used_fallback = False
        index = self.candidate_policy_frames.index(frame)
        safe_values = probabilities[:, index, 0]
        chain_values = probabilities[:, index, 1]
        spread = float(max(np.ptp(safe_values), np.ptp(chain_values)))
        return CausalTransitionDecision(
            accepted=True,
            trigger_policy_frame=frame,
            parent_trigger_policy_frame=self.parent_trigger_policy_frame,
            residual_frames=frame - self.parent_trigger_policy_frame,
            support_distance=support_distance,
            actor_hash=self.actor_hash,
            predicted_safe_probability=float(np.min(safe_values)),
            predicted_chain_probability=float(np.min(chain_values)),
            ensemble_probability_spread=spread,
            used_parent_fallback=used_fallback,
        )

    def _candidate_probabilities(self, observation: NDArray[np.float64]) -> NDArray[np.float64]:
        normalized = (observation - np.asarray(self.feature_center)) / np.asarray(
            self.feature_scale
        )
        head_weights = np.asarray(self.hidden_weights, dtype=np.float64)
        head_bias = np.asarray(self.hidden_bias, dtype=np.float64)
        output_weights = np.asarray(self.output_weights, dtype=np.float64)
        output_bias = np.asarray(self.output_bias, dtype=np.float64)
        result: NDArray[np.float64] = np.empty(
            (4, len(self.candidate_policy_frames), 2), dtype=np.float64
        )
        for candidate_index, frame in enumerate(self.candidate_policy_frames):
            vector = np.concatenate(
                (
                    normalized,
                    np.asarray(
                        ((frame - self.parent_trigger_policy_frame) / self.candidate_frame_scale,),
                        dtype=np.float64,
                    ),
                )
            )
            hidden = np.tanh(np.einsum("hij,j->hi", head_weights, vector) + head_bias)
            logits = np.einsum("hki,hi->hk", output_weights, hidden) + output_bias
            result[:, candidate_index] = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        return result


@dataclass(frozen=True)
class G1CausalSkillTransitionMemoryActor:
    """Local, chance-constrained episodic memory over physics probes."""

    source_snapshot_hash: str
    implementation_hash: str
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    feature_minimum: tuple[float, ...]
    feature_maximum: tuple[float, ...]
    prototype_context_ids: tuple[str, ...]
    prototype_features: tuple[tuple[float, ...], ...]
    safe_labels: tuple[tuple[bool, ...], ...]
    chain_labels: tuple[tuple[bool, ...], ...]
    neighbor_count: int
    minimum_neighbor_chain_fraction: float
    minimum_chain_advantage: float
    maximum_neighbor_distance: float
    candidate_policy_frames: tuple[int, ...] = RISK_CANDIDATE_POLICY_FRAMES
    parent_trigger_policy_frame: int = 88
    maximum_ood_distance: float = 2.0
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    schema_version: str = "rosclaw.growth.causal_transition_memory_actor.v1"

    def __post_init__(self) -> None:
        commitments = (self.source_snapshot_hash, self.implementation_hash)
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        minimum = np.asarray(self.feature_minimum, dtype=np.float64)
        maximum = np.asarray(self.feature_maximum, dtype=np.float64)
        prototypes = np.asarray(self.prototype_features, dtype=np.float64)
        safe = np.asarray(self.safe_labels, dtype=np.bool_)
        chain = np.asarray(self.chain_labels, dtype=np.bool_)
        context_count = len(self.prototype_context_ids)
        if any(
            not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71
            for value in commitments
        ):
            raise ValueError("causal transition memory commitments must be SHA-256 hashes")
        if (
            context_count < 16
            or len(set(self.prototype_context_ids)) != context_count
            or any(not isinstance(value, str) or not value for value in self.prototype_context_ids)
            or center.shape != (_FEATURE_COUNT,)
            or scale.shape != (_FEATURE_COUNT,)
            or minimum.shape != (_FEATURE_COUNT,)
            or maximum.shape != (_FEATURE_COUNT,)
            or prototypes.shape != (context_count, _FEATURE_COUNT)
            or safe.shape != (context_count, len(self.candidate_policy_frames))
            or chain.shape != safe.shape
            or not all(
                np.all(np.isfinite(value))
                for value in (center, scale, minimum, maximum, prototypes)
            )
            or np.any(scale <= 0.0)
            or np.any(minimum > maximum)
            or np.any(chain & ~safe)
            or any(type(value) is not bool for row in self.safe_labels for value in row)
            or any(type(value) is not bool for row in self.chain_labels for value in row)
            or isinstance(self.neighbor_count, bool)
            or not 2 <= self.neighbor_count < context_count
        ):
            raise ValueError("causal transition memory tensors are invalid")
        if (
            tuple(self.candidate_policy_frames) != RISK_CANDIDATE_POLICY_FRAMES
            or self.parent_trigger_policy_frame not in self.candidate_policy_frames
            or not 0.5 <= self.minimum_neighbor_chain_fraction <= 1.0
            or not 0.0 <= self.minimum_chain_advantage <= 0.5
            or not 0.5 <= self.maximum_neighbor_distance <= 20.0
            or not 0.1 <= self.maximum_ood_distance <= 5.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.direct_joint_torque_output
            or self.schema_version != "rosclaw.growth.causal_transition_memory_actor.v1"
        ):
            raise ValueError("causal transition memory actor violates its safety contract")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            **asdict(self),
            "feature_names": list(CAUSAL_TRANSITION_FEATURE_NAMES),
            "algorithm": "chance_constrained_local_episodic_phase_memory_v1",
            "serialized_executable_code": False,
            "pixels_used_for_training": False,
            "output_authority": "BOUNDED_SUCCESSOR_POLICY_PHASE_ONLY",
            "ood_route": "FROZEN_PARENT_TRIGGER",
            "safety_rule": "ALL_K_NEIGHBORS_SAFE",
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(self, features: NDArray[np.float64]) -> CausalTransitionDecision:
        return self._decide(features, excluded_context_id=None)

    def _decide(
        self,
        features: NDArray[np.float64],
        *,
        excluded_context_id: str | None,
    ) -> CausalTransitionDecision:
        observation = np.asarray(features, dtype=np.float64)
        if observation.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(observation)):
            raise ValueError("causal transition memory actor requires fourteen finite features")
        minimum = np.asarray(self.feature_minimum, dtype=np.float64)
        maximum = np.asarray(self.feature_maximum, dtype=np.float64)
        span = np.maximum(maximum - minimum, 0.05)
        box_distance = float(
            np.linalg.norm(
                (np.maximum(minimum - observation, 0.0) + np.maximum(observation - maximum, 0.0))
                / span
            )
        )
        normalized = (observation - np.asarray(self.feature_center)) / np.asarray(
            self.feature_scale
        )
        prototypes = (
            np.asarray(self.prototype_features, dtype=np.float64) - np.asarray(self.feature_center)
        ) / np.asarray(self.feature_scale)
        distances = np.linalg.norm(prototypes - normalized, axis=1)
        if excluded_context_id is not None:
            try:
                distances[self.prototype_context_ids.index(excluded_context_id)] = np.inf
            except ValueError as exc:
                raise ValueError("causal transition memory exclusion is unknown") from exc
        neighbors = np.argsort(distances)[: self.neighbor_count]
        neighbor_distance = float(distances[neighbors[-1]])
        support_distance = max(box_distance, neighbor_distance / self.maximum_neighbor_distance)
        if (
            box_distance > self.maximum_ood_distance
            or not math.isfinite(neighbor_distance)
            or neighbor_distance > self.maximum_neighbor_distance
        ):
            return CausalTransitionDecision(
                accepted=False,
                trigger_policy_frame=self.parent_trigger_policy_frame,
                parent_trigger_policy_frame=self.parent_trigger_policy_frame,
                residual_frames=0,
                support_distance=support_distance,
                actor_hash=self.actor_hash,
                used_parent_fallback=True,
            )
        safe = np.asarray(self.safe_labels, dtype=np.bool_)[neighbors]
        chain = np.asarray(self.chain_labels, dtype=np.bool_)[neighbors]
        parent_index = self.candidate_policy_frames.index(self.parent_trigger_policy_frame)
        parent_fraction = float(np.mean(chain[:, parent_index]))
        choices: list[tuple[float, int, int]] = []
        for index, frame in enumerate(self.candidate_policy_frames):
            if frame == self.parent_trigger_policy_frame:
                continue
            chain_fraction = float(np.mean(chain[:, index]))
            if (
                bool(np.all(safe[:, index]))
                and chain_fraction >= self.minimum_neighbor_chain_fraction
                and chain_fraction >= parent_fraction + self.minimum_chain_advantage
            ):
                choices.append(
                    (chain_fraction, -abs(frame - self.parent_trigger_policy_frame), frame)
                )
        frame = self.parent_trigger_policy_frame
        used_fallback = True
        if choices:
            frame = max(choices)[2]
            used_fallback = False
        index = self.candidate_policy_frames.index(frame)
        return CausalTransitionDecision(
            accepted=True,
            trigger_policy_frame=frame,
            parent_trigger_policy_frame=self.parent_trigger_policy_frame,
            residual_frames=frame - self.parent_trigger_policy_frame,
            support_distance=support_distance,
            actor_hash=self.actor_hash,
            predicted_safe_probability=float(np.mean(safe[:, index])),
            predicted_chain_probability=float(np.mean(chain[:, index])),
            ensemble_probability_spread=None,
            used_parent_fallback=used_fallback,
        )


def save_causal_skill_transition_risk_actor(
    actor: G1CausalSkillTransitionRiskActor, path: Path
) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def save_causal_skill_transition_memory_actor(
    actor: G1CausalSkillTransitionMemoryActor, path: Path
) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_causal_skill_transition_risk_actor(
    path: Path,
) -> G1CausalSkillTransitionRiskActor:
    source = path.expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("causal transition risk actor is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("causal transition risk actor must be an object")
    claimed = payload.pop("actor_hash", None)
    if payload.pop("feature_names", None) != list(CAUSAL_TRANSITION_FEATURE_NAMES):
        raise ValueError("causal transition risk feature contract changed")
    metadata = {
        "algorithm": "four_head_risk_sensitive_discrete_phase_selector_v1",
        "serialized_executable_code": False,
        "pixels_used_for_training": False,
        "output_authority": "BOUNDED_SUCCESSOR_POLICY_PHASE_ONLY",
        "ood_route": "FROZEN_PARENT_TRIGGER",
        "uncertainty_rule": "MINIMUM_PROBABILITY_ACROSS_FOUR_HEADS",
    }
    if any(payload.pop(key, None) != value for key, value in metadata.items()):
        raise ValueError("causal transition risk metadata contract changed")
    for key in (
        "feature_center",
        "feature_scale",
        "feature_minimum",
        "feature_maximum",
        "candidate_policy_frames",
    ):
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(f"causal transition risk actor {key} must be a list")
        payload[key] = tuple(value)
    for key in ("head_training_context_ids", "head_validation_context_ids"):
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(f"causal transition risk actor {key} must be nested lists")
        payload[key] = tuple(tuple(row) for row in value)
    for key in ("hidden_bias", "output_bias"):
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(f"causal transition risk actor {key} must be a matrix")
        payload[key] = tuple(tuple(row) for row in value)
    for key in ("hidden_weights", "output_weights"):
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(f"causal transition risk actor {key} must be a tensor")
        payload[key] = tuple(tuple(tuple(vector) for vector in matrix) for matrix in value)
    try:
        actor = G1CausalSkillTransitionRiskActor(**payload)
    except TypeError as exc:
        raise ValueError("causal transition risk actor payload is invalid") from exc
    if claimed != actor.actor_hash:
        raise ValueError("causal transition risk actor hash does not match")
    return actor


def load_causal_skill_transition_memory_actor(
    path: Path,
) -> G1CausalSkillTransitionMemoryActor:
    source = path.expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("causal transition memory actor is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("causal transition memory actor must be an object")
    claimed = payload.pop("actor_hash", None)
    if payload.pop("feature_names", None) != list(CAUSAL_TRANSITION_FEATURE_NAMES):
        raise ValueError("causal transition memory feature contract changed")
    metadata = {
        "algorithm": "chance_constrained_local_episodic_phase_memory_v1",
        "serialized_executable_code": False,
        "pixels_used_for_training": False,
        "output_authority": "BOUNDED_SUCCESSOR_POLICY_PHASE_ONLY",
        "ood_route": "FROZEN_PARENT_TRIGGER",
        "safety_rule": "ALL_K_NEIGHBORS_SAFE",
    }
    if any(payload.pop(key, None) != value for key, value in metadata.items()):
        raise ValueError("causal transition memory metadata contract changed")
    for key in (
        "feature_center",
        "feature_scale",
        "feature_minimum",
        "feature_maximum",
        "prototype_context_ids",
        "candidate_policy_frames",
    ):
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(f"causal transition memory actor {key} must be a list")
        payload[key] = tuple(value)
    for key in ("prototype_features", "safe_labels", "chain_labels"):
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(f"causal transition memory actor {key} must be a matrix")
        payload[key] = tuple(tuple(row) for row in value)
    try:
        actor = G1CausalSkillTransitionMemoryActor(**payload)
    except TypeError as exc:
        raise ValueError("causal transition memory actor payload is invalid") from exc
    if claimed != actor.actor_hash:
        raise ValueError("causal transition memory actor hash does not match")
    return actor


__all__ = [
    "CausalTransitionProbeSample",
    "G1CausalSkillTransitionRiskActor",
    "G1CausalSkillTransitionMemoryActor",
    "RISK_CANDIDATE_POLICY_FRAMES",
    "load_causal_skill_transition_risk_actor",
    "load_causal_skill_transition_memory_actor",
    "save_causal_skill_transition_memory_actor",
    "save_causal_skill_transition_risk_actor",
]
