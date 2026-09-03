"""Failure-aware runtime selection of pass-to-contact body modes."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.runtime_causal_strike_router import (
    runtime_causal_strike_features,
)
from rosclaw_soccer.sim.contracts import hash_json

_SCHEMA = "rosclaw.growth.g1_runtime_contact_mode_actor.v1"
_FEATURE_COUNT = 7


def _commitment(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{label} must be a SHA-256 commitment")
    return value


@dataclass(frozen=True)
class RuntimeContactModeAction:
    maximum_arrival_advance_frames: int
    stance_offset_x_m: float
    stance_offset_y_m: float = -0.06
    contact_policy_frame: int = 248
    foot_yaw_offset_rad: float = 0.04
    foot_pitch_offset_rad: float = 0.01
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        values = (
            self.stance_offset_x_m,
            self.stance_offset_y_m,
            self.foot_yaw_offset_rad,
            self.foot_pitch_offset_rad,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or self.maximum_arrival_advance_frames not in {0, 12}
            or not -0.12 <= self.stance_offset_x_m <= 0.12
            or not -0.12 <= self.stance_offset_y_m <= 0.12
            or not 238 <= self.contact_policy_frame <= 258
            or not -0.12 <= self.foot_yaw_offset_rad <= 0.12
            or not -0.08 <= self.foot_pitch_offset_rad <= 0.08
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("runtime contact action exceeds its SIM-only envelope")


@dataclass(frozen=True)
class RuntimeContactModeMemory:
    context_hash: str
    trajectory_hash: str
    features: tuple[float, ...]
    action: RuntimeContactModeAction

    def __post_init__(self) -> None:
        _commitment(self.context_hash, "context_hash")
        _commitment(self.trajectory_hash, "trajectory_hash")
        vector = np.asarray(self.features, dtype=np.float64)
        if vector.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(vector)):
            raise ValueError("runtime contact memory features are invalid")


@dataclass(frozen=True)
class RuntimeContactModeDecision:
    accepted: bool
    route: str
    confidence: float
    nearest_success_distance: float
    nearest_same_action_failure_distance: float | None
    selected_context_hash: str | None
    action: RuntimeContactModeAction | None
    actor_hash: str


@dataclass(frozen=True)
class G1RuntimeContactModeActor:
    body_hash: str
    kick_prior_hash: str
    source_discovery_hashes: tuple[str, ...]
    training_snapshot_hash: str
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    successful_memories: tuple[RuntimeContactModeMemory, ...]
    failed_memories: tuple[RuntimeContactModeMemory, ...]
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
            raise ValueError("runtime contact actor needs bound discovery sources")
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
            raise ValueError("runtime contact actor violates its failure-aware SIM-only contract")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            **asdict(self),
            "feature_names": [
                "ball_local_x_m",
                "ball_local_y_m",
                "ball_local_vx_mps",
                "ball_local_vy_mps",
                "ball_arrival_eta_sec",
                "pelvis_local_y_m",
                "joint_velocity_rms_rad_s",
            ],
            "decision_clock": "FIFTH_CONSECUTIVE_MEASURED_INCOMING_BALL_OBSERVATION",
            "algorithm": "failure_aware_nearest_verified_body_contact_mode",
            "stability_plasticity_contract": {
                "stability": "reject OOD or failure-dominated measured arrivals",
                "plasticity": "learn timing and stance only from complete physics trajectories",
            },
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(self, features: Sequence[float]) -> RuntimeContactModeDecision:
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(vector)):
            raise ValueError("runtime contact mode decision features are invalid")
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        normalized = (vector - center) / scale

        def distance(memory: RuntimeContactModeMemory) -> float:
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
            "VERIFIED_RUNTIME_BODY_CONTACT_MODE"
            if accepted
            else (
                "RUNTIME_BODY_CONTACT_FAILURE_FALLBACK"
                if failure_distance is not None
                and success_distance + self.failure_exclusion_margin >= failure_distance
                else "RUNTIME_BODY_CONTACT_OOD_FALLBACK"
            )
        )
        confidence = (
            max(0.0, min(1.0, 1.0 - success_distance / self.maximum_support_distance))
            if accepted
            else 0.0
        )
        return RuntimeContactModeDecision(
            accepted=accepted,
            route=route,
            confidence=confidence,
            nearest_success_distance=success_distance,
            nearest_same_action_failure_distance=failure_distance,
            selected_context_hash=selected.context_hash if accepted else None,
            action=selected.action if accepted else None,
            actor_hash=self.actor_hash,
        )


def save_runtime_contact_mode_actor(actor: G1RuntimeContactModeActor, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_runtime_contact_mode_actor(path: Path) -> G1RuntimeContactModeActor:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime contact mode actor must be an object")
    claimed = payload.pop("actor_hash", None)
    for key in ("feature_names", "decision_clock", "algorithm", "stability_plasticity_contract"):
        payload.pop(key, None)
    payload["source_discovery_hashes"] = tuple(payload["source_discovery_hashes"])
    payload["feature_center"] = tuple(payload["feature_center"])
    payload["feature_scale"] = tuple(payload["feature_scale"])
    for key in ("successful_memories", "failed_memories"):
        payload[key] = tuple(
            RuntimeContactModeMemory(
                context_hash=item["context_hash"],
                trajectory_hash=item["trajectory_hash"],
                features=tuple(item["features"]),
                action=RuntimeContactModeAction(**item["action"]),
            )
            for item in payload[key]
        )
    actor = G1RuntimeContactModeActor(**payload)
    if claimed != actor.actor_hash:
        raise ValueError("runtime contact mode actor hash mismatch")
    return actor


__all__ = [
    "G1RuntimeContactModeActor",
    "RuntimeContactModeAction",
    "RuntimeContactModeDecision",
    "RuntimeContactModeMemory",
    "load_runtime_contact_mode_actor",
    "runtime_causal_strike_features",
    "save_runtime_contact_mode_actor",
]
