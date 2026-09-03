"""Perceptive task-space target selection for a receiving G1 finisher.

The actor runs after a stable incoming ball has been measured.  It selects
only a bounded desired foot velocity.  A separately committed RECEIVE law
owns phase alignment and contact geometry, while the neural contact actor owns
the bounded joint-torque residual.  This separation makes target selection
causally testable and keeps the actor outside direct motor authority.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.runtime_receive_actor import (
    RUNTIME_RECEIVE_FEATURE_NAMES,
    RuntimeReceiveAction,
)
from rosclaw_soccer.sim.contracts import hash_json

_FEATURE_COUNT = len(RUNTIME_RECEIVE_FEATURE_NAMES)
_SCHEMA = "rosclaw_soccer.g1_runtime_contact_target_actor.v1"


def _commitment(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{label} must be a SHA-256 commitment")
    return value


@dataclass(frozen=True)
class RuntimeContactTargetAction:
    """A bounded task-space request; it is not a joint or torque command."""

    target_foot_velocity_xyz_mps: tuple[float, float, float]
    activation_ceiling: str = "SIM_ONLY"
    direct_joint_torque_output: bool = False

    def __post_init__(self) -> None:
        target = np.asarray(self.target_foot_velocity_xyz_mps, dtype=np.float64)
        if (
            target.shape != (3,)
            or not np.all(np.isfinite(target))
            or not 5.0 <= target[0] <= 12.0
            or not -6.0 <= target[1] <= 6.0
            or not -3.0 <= target[2] <= 6.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.direct_joint_torque_output
        ):
            raise ValueError("runtime contact target exceeds its SIM-only task envelope")

    @property
    def action_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class RuntimeContactTargetMemory:
    context_hash: str
    trajectory_hash: str
    features: tuple[float, ...]
    action: RuntimeContactTargetAction
    quality_score: float

    def __post_init__(self) -> None:
        _commitment(self.context_hash, "context_hash")
        _commitment(self.trajectory_hash, "trajectory_hash")
        vector = np.asarray(self.features, dtype=np.float64)
        if (
            vector.shape != (_FEATURE_COUNT,)
            or not np.all(np.isfinite(vector))
            or not math.isfinite(self.quality_score)
            or not 0.0 <= self.quality_score <= 12.0
        ):
            raise ValueError("runtime contact target memory is invalid")


@dataclass(frozen=True)
class RuntimeContactTargetDecision:
    accepted: bool
    route: str
    confidence: float
    nearest_success_distance: float
    nearest_same_action_failure_distance: float | None
    selected_context_hash: str | None
    action: RuntimeContactTargetAction | None
    actor_hash: str


@dataclass(frozen=True)
class G1RuntimeContactTargetActor:
    """Failure-aware target selector owned by ``red.finisher.first_touch``."""

    body_hash: str
    kick_prior_hash: str
    roster_hash: str
    finisher_self_model_hash: str
    neural_contact_actor_hash: str
    source_evidence_hashes: tuple[str, ...]
    training_snapshot_hash: str
    required_receive_action: RuntimeReceiveAction
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    successful_memories: tuple[RuntimeContactTargetMemory, ...]
    failed_memories: tuple[RuntimeContactTargetMemory, ...]
    maximum_support_distance: float = 2.75
    failure_exclusion_margin: float = 0.05
    quality_tie_tolerance: float = 1.0e-9
    agent_id: str = "red.finisher"
    primary_role: str = "finisher"
    tactical_intent: str = "receive_and_strike"
    owned_skill: str = "contact_target_selection"
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    online_hot_swap_allowed: bool = False
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        for value, label in (
            (self.body_hash, "body_hash"),
            (self.kick_prior_hash, "kick_prior_hash"),
            (self.roster_hash, "roster_hash"),
            (self.finisher_self_model_hash, "finisher_self_model_hash"),
            (self.neural_contact_actor_hash, "neural_contact_actor_hash"),
            (self.training_snapshot_hash, "training_snapshot_hash"),
        ):
            _commitment(value, label)
        if not self.source_evidence_hashes:
            raise ValueError("runtime contact target actor needs evidence commitments")
        for index, value in enumerate(self.source_evidence_hashes):
            _commitment(value, f"source_evidence_hashes[{index}]")
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        success_hashes = {memory.trajectory_hash for memory in self.successful_memories}
        failure_hashes = {memory.trajectory_hash for memory in self.failed_memories}
        if (
            self.schema_version != _SCHEMA
            or center.shape != (_FEATURE_COUNT,)
            or scale.shape != (_FEATURE_COUNT,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
            or len(self.successful_memories) < 4
            or len(self.failed_memories) < 2
            or success_hashes & failure_hashes
            or not 0.25 <= self.maximum_support_distance <= 4.0
            or not 0.0 <= self.failure_exclusion_margin <= 0.50
            or not 0.0 <= self.quality_tie_tolerance <= 1.0e-6
            or (self.agent_id, self.primary_role, self.tactical_intent, self.owned_skill)
            != (
                "red.finisher",
                "finisher",
                "receive_and_strike",
                "contact_target_selection",
            )
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.direct_joint_torque_output
            or self.online_hot_swap_allowed
        ):
            raise ValueError("runtime contact target actor violates its role-bound contract")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "feature_names": list(RUNTIME_RECEIVE_FEATURE_NAMES),
            "algorithm": "failure_aware_nearest_verified_task_target_with_quality_tie_break",
            "decision_clock": "FIFTH_CONSECUTIVE_MEASURED_INCOMING_BALL_OBSERVATION",
            "authority": "TASK_SPACE_TARGET_VELOCITY_ONLY",
            "joint_torque_owner": "content_bound_neural_contact_actor",
            "stability_plasticity_contract": {
                "stability": "immutable latch, OOD and same-action failure exclusion",
                "plasticity": "new complete CPU MuJoCo evidence creates a new artifact",
            },
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(self, features: Sequence[float]) -> RuntimeContactTargetDecision:
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(vector)):
            raise ValueError("runtime contact target features are invalid")
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        normalized = (vector - center) / scale

        def distance(memory: RuntimeContactTargetMemory) -> float:
            candidate = (np.asarray(memory.features, dtype=np.float64) - center) / scale
            return float(np.linalg.norm(candidate - normalized))

        measured = [(distance(memory), memory) for memory in self.successful_memories]
        minimum_distance = min(value for value, _ in measured)
        local = [
            item for item in measured if item[0] <= minimum_distance + self.quality_tie_tolerance
        ]
        success_distance, selected = sorted(
            local,
            key=lambda item: (
                -item[1].quality_score,
                item[1].action.action_hash,
                item[1].trajectory_hash,
            ),
        )[0]
        same_action_failures = sorted(
            distance(memory) for memory in self.failed_memories if memory.action == selected.action
        )
        failure_distance = same_action_failures[0] if same_action_failures else None
        accepted = bool(
            success_distance <= self.maximum_support_distance
            and (
                failure_distance is None
                or success_distance + self.failure_exclusion_margin < failure_distance
            )
        )
        route = (
            "VERIFIED_PERCEPTIVE_CONTACT_TARGET"
            if accepted
            else (
                "CONTACT_TARGET_OOD_FALLBACK"
                if success_distance > self.maximum_support_distance
                else "CONTACT_TARGET_FAILURE_MEMORY_FALLBACK"
            )
        )
        return RuntimeContactTargetDecision(
            accepted=accepted,
            route=route,
            confidence=(
                max(0.0, min(1.0, 1.0 - success_distance / self.maximum_support_distance))
                if accepted
                else 0.0
            ),
            nearest_success_distance=success_distance,
            nearest_same_action_failure_distance=failure_distance,
            selected_context_hash=selected.context_hash if accepted else None,
            action=selected.action if accepted else None,
            actor_hash=self.actor_hash,
        )


def save_runtime_contact_target_actor(actor: G1RuntimeContactTargetActor, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_runtime_contact_target_actor(path: Path) -> G1RuntimeContactTargetActor:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime contact target actor must be an object")
    claimed = payload.pop("actor_hash", None)
    for key in (
        "feature_names",
        "algorithm",
        "decision_clock",
        "authority",
        "joint_torque_owner",
        "stability_plasticity_contract",
    ):
        payload.pop(key, None)
    payload["source_evidence_hashes"] = tuple(payload["source_evidence_hashes"])
    payload["feature_center"] = tuple(payload["feature_center"])
    payload["feature_scale"] = tuple(payload["feature_scale"])
    payload["required_receive_action"] = RuntimeReceiveAction(**payload["required_receive_action"])
    for key in ("successful_memories", "failed_memories"):
        payload[key] = tuple(
            RuntimeContactTargetMemory(
                context_hash=item["context_hash"],
                trajectory_hash=item["trajectory_hash"],
                features=tuple(item["features"]),
                action=RuntimeContactTargetAction(
                    target_foot_velocity_xyz_mps=tuple(
                        item["action"]["target_foot_velocity_xyz_mps"]
                    ),
                    activation_ceiling=item["action"]["activation_ceiling"],
                    direct_joint_torque_output=item["action"]["direct_joint_torque_output"],
                ),
                quality_score=float(item["quality_score"]),
            )
            for item in payload[key]
        )
    actor = G1RuntimeContactTargetActor(**payload)
    if claimed != actor.actor_hash:
        raise ValueError("runtime contact target actor hash mismatch")
    return actor


__all__ = [
    "G1RuntimeContactTargetActor",
    "RuntimeContactTargetAction",
    "RuntimeContactTargetDecision",
    "RuntimeContactTargetMemory",
    "load_runtime_contact_target_actor",
    "save_runtime_contact_target_actor",
]
