"""Failure-aware routing between bounded pass-to-strike contact modes."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json

_SCHEMA = "rosclaw.growth.g1_failure_aware_causal_strike_router.v1"
_FEATURE_COUNT = 5


def _commitment(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{label} must be a SHA-256 commitment")
    return value


@dataclass(frozen=True)
class CausalStrikeRouteAction:
    maximum_arrival_advance_frames: int
    foot_yaw_offset_rad: float = 0.04
    upper_corner_muscle_memory: bool = True
    shooter_precontact_joint_guard: bool = True
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_arrival_advance_frames, bool)
            or not 0 <= self.maximum_arrival_advance_frames <= 12
            or not math.isfinite(self.foot_yaw_offset_rad)
            or not -0.12 <= self.foot_yaw_offset_rad <= 0.12
            or not self.upper_corner_muscle_memory
            or not self.shooter_precontact_joint_guard
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("causal strike route action exceeds its safe option envelope")


@dataclass(frozen=True)
class CausalStrikeRouteMemory:
    context_hash: str
    trajectory_hash: str
    features: tuple[float, ...]
    action: CausalStrikeRouteAction

    def __post_init__(self) -> None:
        _commitment(self.context_hash, "context_hash")
        _commitment(self.trajectory_hash, "trajectory_hash")
        vector = np.asarray(self.features, dtype=np.float64)
        if vector.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(vector)):
            raise ValueError("causal strike route memory features are invalid")


@dataclass(frozen=True)
class CausalStrikeRouteDecision:
    accepted: bool
    route: str
    confidence: float
    nearest_success_distance: float
    nearest_same_action_failure_distance: float | None
    selected_context_hash: str | None
    action: CausalStrikeRouteAction | None
    actor_hash: str


@dataclass(frozen=True)
class G1FailureAwareCausalStrikeRouter:
    body_hash: str
    kick_prior_hash: str
    source_discovery_hash: str
    training_snapshot_hash: str
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    successful_memories: tuple[CausalStrikeRouteMemory, ...]
    failed_memories: tuple[CausalStrikeRouteMemory, ...]
    maximum_support_distance: float = 2.25
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
            (self.source_discovery_hash, "source_discovery_hash"),
            (self.training_snapshot_hash, "training_snapshot_hash"),
        ):
            _commitment(value, label)
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
            or {item.trajectory_hash for item in self.successful_memories}
            & {item.trajectory_hash for item in self.failed_memories}
            or not math.isfinite(self.maximum_support_distance)
            or not 0.25 <= self.maximum_support_distance <= 4.0
            or not math.isfinite(self.failure_exclusion_margin)
            or not 0.0 <= self.failure_exclusion_margin <= 0.50
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.online_hot_swap_allowed
        ):
            raise ValueError("causal strike router violates its failure-aware SIM-only contract")

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
            "algorithm": "failure_aware_nearest_verified_contact_mode",
            "direct_joint_torque_output": False,
            "pixels_used_for_training": False,
            "stability_plasticity_contract": {
                "stability": "reject OOD or failure-dominated routes",
                "plasticity": "append physically verified contact modes and retained failures",
            },
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(self, features: Sequence[float]) -> CausalStrikeRouteDecision:
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(vector)):
            raise ValueError("causal strike route decision features are invalid")
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        normalized = (vector - center) / scale

        def distance(memory: CausalStrikeRouteMemory) -> float:
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
            "VERIFIED_CAUSAL_STRIKE_MODE"
            if accepted
            else (
                "FAILURE_MEMORY_FALLBACK"
                if failure_distance is not None
                and success_distance + self.failure_exclusion_margin >= failure_distance
                else "OUT_OF_SUPPORT_FALLBACK"
            )
        )
        confidence = (
            max(0.0, min(1.0, 1.0 - success_distance / self.maximum_support_distance))
            if accepted
            else 0.0
        )
        return CausalStrikeRouteDecision(
            accepted=accepted,
            route=route,
            confidence=confidence,
            nearest_success_distance=success_distance,
            nearest_same_action_failure_distance=failure_distance,
            selected_context_hash=selected.context_hash if accepted else None,
            action=selected.action if accepted else None,
            actor_hash=self.actor_hash,
        )


def save_causal_strike_router(actor: G1FailureAwareCausalStrikeRouter, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_causal_strike_router(path: Path) -> G1FailureAwareCausalStrikeRouter:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("causal strike router artifact must be an object")
    claimed = payload.pop("actor_hash", None)
    for key in (
        "feature_names",
        "algorithm",
        "direct_joint_torque_output",
        "pixels_used_for_training",
        "stability_plasticity_contract",
    ):
        payload.pop(key, None)
    payload["feature_center"] = tuple(payload["feature_center"])
    payload["feature_scale"] = tuple(payload["feature_scale"])
    for key in ("successful_memories", "failed_memories"):
        payload[key] = tuple(
            CausalStrikeRouteMemory(
                context_hash=item["context_hash"],
                trajectory_hash=item["trajectory_hash"],
                features=tuple(item["features"]),
                action=CausalStrikeRouteAction(**item["action"]),
            )
            for item in payload[key]
        )
    actor = G1FailureAwareCausalStrikeRouter(**payload)
    if claimed != actor.actor_hash:
        raise ValueError("causal strike router hash does not match its payload")
    return actor


__all__ = [
    "CausalStrikeRouteAction",
    "CausalStrikeRouteDecision",
    "CausalStrikeRouteMemory",
    "G1FailureAwareCausalStrikeRouter",
    "load_causal_strike_router",
    "save_causal_strike_router",
]
