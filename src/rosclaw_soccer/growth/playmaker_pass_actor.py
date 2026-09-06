"""Failure-aware, context-conditioned playmaker pass muscle memory."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.causal_transition_growth import CausalTransitionContext
from rosclaw_soccer.training.playmaker_pass_discovery import PlaymakerPassProbeAction

_SCHEMA = "rosclaw_soccer.g1_playmaker_pass_actor.v1"
_FEATURE_COUNT = 5


def playmaker_pass_features(context: CausalTransitionContext) -> tuple[float, ...]:
    """Expose only task state available before a rollout begins."""

    return (
        context.receiver_lane_m,
        context.reception_target_x_m,
        context.passer_ball_local_xy_m[0],
        context.passer_ball_local_xy_m[1],
        context.ball_ground_friction,
    )


@dataclass(frozen=True)
class PlaymakerPassMemory:
    context_hash: str
    trajectory_hash: str
    features: tuple[float, ...]
    action: PlaymakerPassProbeAction
    delivery_error_m: float
    safe: bool
    ordered_contacts: bool
    schema_version: str = "rosclaw_soccer.playmaker_pass_memory.v1"

    def __post_init__(self) -> None:
        vector = np.asarray(self.features, dtype=np.float64)
        if (
            not _is_hash(self.context_hash)
            or not _is_hash(self.trajectory_hash)
            or vector.shape != (_FEATURE_COUNT,)
            or not np.all(np.isfinite(vector))
            or not math.isfinite(self.delivery_error_m)
            or not 0.0 <= self.delivery_error_m <= 5.0
        ):
            raise ValueError("playmaker pass memory is invalid")

    @property
    def memory_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlaymakerPassDecision:
    accepted: bool
    route: str
    confidence: float
    nearest_success_distance: float
    nearest_same_action_failure_distance: float | None
    selected_context_hash: str | None
    action: PlaymakerPassProbeAction | None
    actor_hash: str


@dataclass(frozen=True)
class G1PlaymakerPassActor:
    body_hash: str
    source_discovery_hash: str
    source_holdout_hash: str
    frozen_finisher_actor_hash: str
    frozen_goalkeeper_policy_hash: str
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    successful_memories: tuple[PlaymakerPassMemory, ...]
    failed_memories: tuple[PlaymakerPassMemory, ...]
    maximum_support_distance: float = 2.25
    failure_exclusion_margin: float = 0.05
    maximum_delivery_error_m: float = 0.45
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    online_hot_swap_allowed: bool = False
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        commitments = (
            self.body_hash,
            self.source_discovery_hash,
            self.source_holdout_hash,
            self.frozen_finisher_actor_hash,
            self.frozen_goalkeeper_policy_hash,
        )
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        if (
            any(not _is_hash(value) for value in commitments)
            or center.shape != (_FEATURE_COUNT,)
            or scale.shape != (_FEATURE_COUNT,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
            or len(self.successful_memories) < 6
            or len(self.failed_memories) < 8
            or len({value.memory_hash for value in self.successful_memories})
            != len(self.successful_memories)
            or len({value.memory_hash for value in self.failed_memories})
            != len(self.failed_memories)
            or not 0.50 <= self.maximum_support_distance <= 4.0
            or not 0.0 <= self.failure_exclusion_margin <= 0.50
            or not 0.10 <= self.maximum_delivery_error_m <= 0.45
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.online_hot_swap_allowed
            or self.schema_version != _SCHEMA
        ):
            raise ValueError("playmaker pass actor violates its frozen SIM-only contract")
        if any(
            not memory.safe
            or not memory.ordered_contacts
            or memory.delivery_error_m > self.maximum_delivery_error_m
            for memory in self.successful_memories
        ):
            raise ValueError("playmaker success memory contains an unqualified pass")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "feature_names": [
                "receiver_lane_m",
                "reception_target_x_m",
                "passer_ball_local_x_m",
                "passer_ball_local_y_m",
                "ball_ground_friction",
            ],
            "algorithm": "nearest_verified_pass_with_same_action_failure_exclusion",
            "decision_clock": "PRE_ROLLOUT_ROLE_PLAN",
            "plastic_role": "playmaker",
            "frozen_roles": ["finisher", "goalkeeper"],
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(self, features: Sequence[float]) -> PlaymakerPassDecision:
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(vector)):
            raise ValueError("playmaker pass features are invalid")
        scale = np.asarray(self.feature_scale, dtype=np.float64)

        def distance(memory: PlaymakerPassMemory) -> float:
            candidate = np.asarray(memory.features, dtype=np.float64)
            return float(np.linalg.norm((candidate - vector) / scale))

        ranked = sorted(
            ((distance(memory), memory) for memory in self.successful_memories),
            key=lambda item: (
                item[0],
                item[1].delivery_error_m,
                item[1].action.action_hash,
                item[1].trajectory_hash,
            ),
        )
        success_distance, selected = ranked[0]
        failure_distances = sorted(
            distance(memory) for memory in self.failed_memories if memory.action == selected.action
        )
        failure_distance = failure_distances[0] if failure_distances else None
        accepted = bool(
            success_distance <= self.maximum_support_distance
            and (
                failure_distance is None
                or success_distance + self.failure_exclusion_margin < failure_distance
            )
        )
        return PlaymakerPassDecision(
            accepted=accepted,
            route=(
                "VERIFIED_ROLE_LOCAL_PASS"
                if accepted
                else (
                    "SAME_ACTION_FAILURE_FALLBACK"
                    if failure_distance is not None
                    else "OUT_OF_DISTRIBUTION_FALLBACK"
                )
            ),
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


def save_playmaker_pass_actor(actor: G1PlaymakerPassActor, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_playmaker_pass_actor(path: Path) -> G1PlaymakerPassActor:
    source = path.expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("playmaker pass actor must be a JSON object")
    payload = cast(dict[str, Any], value)
    claimed = payload.pop("actor_hash", None)
    payload.pop("feature_names", None)
    payload.pop("algorithm", None)
    payload.pop("decision_clock", None)
    payload.pop("plastic_role", None)
    payload.pop("frozen_roles", None)
    payload["successful_memories"] = tuple(
        _memory_from_dict(item) for item in payload["successful_memories"]
    )
    payload["failed_memories"] = tuple(
        _memory_from_dict(item) for item in payload["failed_memories"]
    )
    payload["feature_center"] = tuple(payload["feature_center"])
    payload["feature_scale"] = tuple(payload["feature_scale"])
    actor = G1PlaymakerPassActor(**payload)
    if claimed != actor.actor_hash:
        raise ValueError("playmaker pass actor content hash changed")
    return actor


def _memory_from_dict(value: dict[str, Any]) -> PlaymakerPassMemory:
    payload = dict(value)
    payload["features"] = tuple(payload["features"])
    payload["action"] = PlaymakerPassProbeAction(**payload["action"])
    return PlaymakerPassMemory(**payload)


def _is_hash(value: str) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) == 71


__all__ = [
    "G1PlaymakerPassActor",
    "PlaymakerPassDecision",
    "PlaymakerPassMemory",
    "load_playmaker_pass_actor",
    "playmaker_pass_features",
    "save_playmaker_pass_actor",
]
