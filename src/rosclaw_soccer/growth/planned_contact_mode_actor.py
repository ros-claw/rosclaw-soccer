"""Failure-aware pre-rollout planning for whole-body contact modes."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.runtime_contact_mode_actor import RuntimeContactModeAction
from rosclaw_soccer.sim.contracts import hash_json

_SCHEMA = "rosclaw.growth.g1_planned_contact_mode_actor.v1"
_FEATURE_COUNT = 5


def planned_contact_mode_features(
    *,
    receiver_lane_m: float,
    reception_target_x_m: float,
    passer_ball_local_xy_m: Sequence[float],
    ball_ground_friction: float,
) -> tuple[float, ...]:
    ball = np.asarray(passer_ball_local_xy_m, dtype=np.float64)
    values = np.asarray(
        (
            receiver_lane_m,
            reception_target_x_m,
            *(ball.tolist() if ball.shape == (2,) else (math.nan, math.nan)),
            ball_ground_friction,
        ),
        dtype=np.float64,
    )
    if values.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(values)):
        raise ValueError("planned contact features must be five finite values")
    return tuple(float(item) for item in values)


def _commitment(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{label} must be a SHA-256 commitment")
    return value


@dataclass(frozen=True)
class PlannedContactModeMemory:
    context_hash: str
    trajectory_hash: str
    features: tuple[float, ...]
    action: RuntimeContactModeAction

    def __post_init__(self) -> None:
        _commitment(self.context_hash, "context_hash")
        _commitment(self.trajectory_hash, "trajectory_hash")
        vector = np.asarray(self.features, dtype=np.float64)
        if vector.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(vector)):
            raise ValueError("planned contact memory features are invalid")


@dataclass(frozen=True)
class PlannedContactModeDecision:
    accepted: bool
    route: str
    confidence: float
    nearest_success_distance: float
    nearest_same_action_failure_distance: float | None
    selected_context_hash: str | None
    action: RuntimeContactModeAction | None
    actor_hash: str


@dataclass(frozen=True)
class G1PlannedContactModeActor:
    body_hash: str
    kick_prior_hash: str
    source_discovery_hashes: tuple[str, ...]
    training_snapshot_hash: str
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    successful_memories: tuple[PlannedContactModeMemory, ...]
    failed_memories: tuple[PlannedContactModeMemory, ...]
    maximum_support_distance: float = 1.75
    failure_exclusion_margin: float = 0.05
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    online_hot_swap_allowed: bool = False
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        for value, label in (
            (self.body_hash, "body_hash"),
            (self.kick_prior_hash, "kick_prior_hash"),
            (self.training_snapshot_hash, "training_snapshot_hash"),
        ):
            _commitment(value, label)
        if not self.source_discovery_hashes:
            raise ValueError("planned contact actor needs discovery sources")
        for index, value in enumerate(self.source_discovery_hashes):
            _commitment(value, f"source_discovery_hashes[{index}]")
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        if (
            self.schema_version != _SCHEMA
            or center.shape != (_FEATURE_COUNT,)
            or scale.shape != (_FEATURE_COUNT,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
            or len(self.successful_memories) < 4
            or len(self.failed_memories) < 4
            or not math.isfinite(self.maximum_support_distance)
            or not 0.25 <= self.maximum_support_distance <= 4.0
            or not math.isfinite(self.failure_exclusion_margin)
            or not 0.0 <= self.failure_exclusion_margin <= 0.50
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.online_hot_swap_allowed
        ):
            raise ValueError("planned contact actor violates its failure-aware SIM-only contract")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            **asdict(self),
            "feature_names": [
                "receiver_lane_m",
                "reception_target_x_m",
                "passer_ball_local_x_m",
                "passer_ball_local_y_m",
                "ball_ground_friction",
            ],
            "decision_clock": "PRE_ROLLOUT_TASK_PLAN",
            "algorithm": "failure_aware_nearest_verified_precontact_mode",
            "direct_joint_torque_output": False,
            "stability_plasticity_contract": {
                "stability": "abstain outside verified plan support",
                "plasticity": "learn coarse stance/timing modes from full physics outcomes",
            },
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(self, features: Sequence[float]) -> PlannedContactModeDecision:
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(vector)):
            raise ValueError("planned contact decision features are invalid")
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        normalized = (vector - center) / scale

        def distance(memory: PlannedContactModeMemory) -> float:
            candidate = (np.asarray(memory.features, dtype=np.float64) - center) / scale
            return float(np.linalg.norm(candidate - normalized))

        successes = sorted(
            ((distance(memory), memory) for memory in self.successful_memories),
            key=lambda item: (item[0], item[1].context_hash, item[1].trajectory_hash),
        )
        success_distance, selected = successes[0]
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
            "VERIFIED_PRECONTACT_BODY_MODE"
            if accepted
            else (
                "PRECONTACT_FAILURE_MEMORY_FALLBACK"
                if failure_distance is not None
                and success_distance + self.failure_exclusion_margin >= failure_distance
                else "PRECONTACT_OOD_FALLBACK"
            )
        )
        confidence = (
            max(0.0, min(1.0, 1.0 - success_distance / self.maximum_support_distance))
            if accepted
            else 0.0
        )
        return PlannedContactModeDecision(
            accepted=accepted,
            route=route,
            confidence=confidence,
            nearest_success_distance=success_distance,
            nearest_same_action_failure_distance=failure_distance,
            selected_context_hash=selected.context_hash if accepted else None,
            action=selected.action if accepted else None,
            actor_hash=self.actor_hash,
        )


def save_planned_contact_mode_actor(actor: G1PlannedContactModeActor, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_planned_contact_mode_actor(path: Path) -> G1PlannedContactModeActor:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("planned contact mode actor must be an object")
    claimed = payload.pop("actor_hash", None)
    for key in (
        "feature_names",
        "decision_clock",
        "algorithm",
        "direct_joint_torque_output",
        "stability_plasticity_contract",
    ):
        payload.pop(key, None)
    payload["source_discovery_hashes"] = tuple(payload["source_discovery_hashes"])
    payload["feature_center"] = tuple(payload["feature_center"])
    payload["feature_scale"] = tuple(payload["feature_scale"])
    for key in ("successful_memories", "failed_memories"):
        payload[key] = tuple(
            PlannedContactModeMemory(
                context_hash=item["context_hash"],
                trajectory_hash=item["trajectory_hash"],
                features=tuple(item["features"]),
                action=RuntimeContactModeAction(**item["action"]),
            )
            for item in payload[key]
        )
    actor = G1PlannedContactModeActor(**payload)
    if claimed != actor.actor_hash:
        raise ValueError("planned contact mode actor hash mismatch")
    return actor


__all__ = [
    "G1PlannedContactModeActor",
    "PlannedContactModeDecision",
    "PlannedContactModeMemory",
    "load_planned_contact_mode_actor",
    "planned_contact_mode_features",
    "save_planned_contact_mode_actor",
]
