"""Failure-aware runtime routing from measured pass arrival state."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.causal_strike_router import (
    CausalStrikeRouteAction,
    CausalStrikeRouteDecision,
)
from rosclaw_soccer.sim.contracts import hash_json

_SCHEMA = "rosclaw.growth.g1_runtime_causal_strike_router.v1"
_FEATURE_NAMES = (
    "ball_local_x_m",
    "ball_local_y_m",
    "ball_local_vx_mps",
    "ball_local_vy_mps",
    "ball_arrival_eta_sec",
    "pelvis_local_y_m",
    "joint_velocity_rms_rad_s",
)
_FEATURE_COUNT = len(_FEATURE_NAMES)


def runtime_causal_strike_features(
    *,
    ball_local_position_m: Sequence[float],
    ball_local_velocity_mps: Sequence[float],
    ball_arrival_eta_sec: float,
    pelvis_local_position_m: Sequence[float],
    joint_velocity_rad_s: Sequence[float],
) -> tuple[float, ...]:
    """Build the online-only route observation used by training and runtime."""

    ball_position = np.asarray(ball_local_position_m, dtype=np.float64)
    ball_velocity = np.asarray(ball_local_velocity_mps, dtype=np.float64)
    pelvis_position = np.asarray(pelvis_local_position_m, dtype=np.float64)
    joint_velocity = np.asarray(joint_velocity_rad_s, dtype=np.float64)
    if (
        ball_position.shape != (3,)
        or ball_velocity.shape != (3,)
        or pelvis_position.shape != (3,)
        or joint_velocity.shape != (29,)
        or not all(
            np.all(np.isfinite(value))
            for value in (ball_position, ball_velocity, pelvis_position, joint_velocity)
        )
        or not math.isfinite(ball_arrival_eta_sec)
        or ball_arrival_eta_sec < 0.0
    ):
        raise ValueError("runtime causal strike features must be finite G1 observations")
    return (
        float(ball_position[0]),
        float(ball_position[1]),
        float(ball_velocity[0]),
        float(ball_velocity[1]),
        float(ball_arrival_eta_sec),
        float(pelvis_position[1]),
        float(np.sqrt(np.mean(np.square(joint_velocity)))),
    )


def _commitment(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{label} must be a SHA-256 commitment")
    return value


@dataclass(frozen=True)
class RuntimeCausalStrikeMemory:
    context_hash: str
    trajectory_hash: str
    features: tuple[float, ...]
    action: CausalStrikeRouteAction

    def __post_init__(self) -> None:
        _commitment(self.context_hash, "context_hash")
        _commitment(self.trajectory_hash, "trajectory_hash")
        vector = np.asarray(self.features, dtype=np.float64)
        if vector.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(vector)):
            raise ValueError("runtime causal strike memory features are invalid")


@dataclass(frozen=True)
class G1RuntimeCausalStrikeRouter:
    body_hash: str
    kick_prior_hash: str
    source_discovery_hashes: tuple[str, ...]
    training_snapshot_hash: str
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    successful_memories: tuple[RuntimeCausalStrikeMemory, ...]
    failed_memories: tuple[RuntimeCausalStrikeMemory, ...]
    maximum_support_distance: float = 2.50
    failure_exclusion_margin: float = 0.05
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    online_hot_swap_allowed: bool = False
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        _commitment(self.body_hash, "body_hash")
        _commitment(self.kick_prior_hash, "kick_prior_hash")
        _commitment(self.training_snapshot_hash, "training_snapshot_hash")
        if not self.source_discovery_hashes:
            raise ValueError("runtime strike router needs bound discovery sources")
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
            or {item.trajectory_hash for item in self.successful_memories}
            & {item.trajectory_hash for item in self.failed_memories}
            or not math.isfinite(self.maximum_support_distance)
            or not 0.25 <= self.maximum_support_distance <= 4.0
            or not math.isfinite(self.failure_exclusion_margin)
            or not 0.0 <= self.failure_exclusion_margin <= 0.50
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.direct_joint_torque_output
            or self.online_hot_swap_allowed
        ):
            raise ValueError("runtime strike router violates its SIM-only contract")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            **asdict(self),
            "feature_names": list(_FEATURE_NAMES),
            "decision_clock": "FIFTH_CONSECUTIVE_MEASURED_INCOMING_BALL_OBSERVATION",
            "algorithm": "failure_aware_nearest_verified_runtime_contact_mode",
            "stability_plasticity_contract": {
                "stability": "reject OOD or failure-dominated measured arrivals",
                "plasticity": "append complete physics trajectories including failures",
            },
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(self, features: Sequence[float]) -> CausalStrikeRouteDecision:
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(vector)):
            raise ValueError("runtime causal strike decision features are invalid")
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        normalized = (vector - center) / scale

        def distance(memory: RuntimeCausalStrikeMemory) -> float:
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
            "VERIFIED_RUNTIME_CAUSAL_STRIKE_MODE"
            if accepted
            else (
                "RUNTIME_FAILURE_MEMORY_FALLBACK"
                if failure_distance is not None
                and success_distance + self.failure_exclusion_margin >= failure_distance
                else "RUNTIME_OUT_OF_SUPPORT_FALLBACK"
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


def save_runtime_causal_strike_router(actor: G1RuntimeCausalStrikeRouter, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_runtime_causal_strike_router(path: Path) -> G1RuntimeCausalStrikeRouter:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime causal strike router artifact must be an object")
    claimed = payload.pop("actor_hash", None)
    for key in ("feature_names", "decision_clock", "algorithm", "stability_plasticity_contract"):
        payload.pop(key, None)
    payload["source_discovery_hashes"] = tuple(payload["source_discovery_hashes"])
    payload["feature_center"] = tuple(payload["feature_center"])
    payload["feature_scale"] = tuple(payload["feature_scale"])
    for key in ("successful_memories", "failed_memories"):
        payload[key] = tuple(
            RuntimeCausalStrikeMemory(
                context_hash=item["context_hash"],
                trajectory_hash=item["trajectory_hash"],
                features=tuple(item["features"]),
                action=CausalStrikeRouteAction(**item["action"]),
            )
            for item in payload[key]
        )
    actor = G1RuntimeCausalStrikeRouter(**payload)
    if claimed != actor.actor_hash:
        raise ValueError("runtime causal strike router hash does not match its payload")
    return actor


__all__ = [
    "G1RuntimeCausalStrikeRouter",
    "RuntimeCausalStrikeMemory",
    "load_runtime_causal_strike_router",
    "runtime_causal_strike_features",
    "save_runtime_causal_strike_router",
]
