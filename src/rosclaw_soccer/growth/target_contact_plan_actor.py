"""Failure-aware pre-rollout planning for target-conditioned contact."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.planned_contact_mode_actor import planned_contact_mode_features
from rosclaw_soccer.sim.contracts import hash_json

_SCHEMA = "rosclaw.growth.g1_target_contact_plan_actor.v1"
_FEATURE_COUNT = 5


def _commitment(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{label} must be a SHA-256 commitment")
    return value


@dataclass(frozen=True)
class TargetContactPlanAction:
    maximum_arrival_advance_frames: int
    stance_offset_x_m: float
    stance_offset_y_m: float
    contact_policy_frame: int
    foot_yaw_offset_rad: float
    foot_pitch_offset_rad: float
    target_foot_velocity_xyz_mps: tuple[float, float, float]
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.stance_offset_x_m,
                self.stance_offset_y_m,
                self.foot_yaw_offset_rad,
                self.foot_pitch_offset_rad,
                *self.target_foot_velocity_xyz_mps,
            ),
            dtype=np.float64,
        )
        if (
            values.shape != (7,)
            or not np.all(np.isfinite(values))
            or self.maximum_arrival_advance_frames not in {0, 6, 12, 18}
            or not -0.16 <= self.stance_offset_x_m <= 0.16
            or not -0.16 <= self.stance_offset_y_m <= 0.16
            or not 238 <= self.contact_policy_frame <= 258
            or not -0.12 <= self.foot_yaw_offset_rad <= 0.12
            or not -0.08 <= self.foot_pitch_offset_rad <= 0.08
            or not 5.0 <= self.target_foot_velocity_xyz_mps[0] <= 12.0
            or not -6.0 <= self.target_foot_velocity_xyz_mps[1] <= 6.0
            or not -3.0 <= self.target_foot_velocity_xyz_mps[2] <= 6.0
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("target contact plan action exceeds its SIM-only envelope")

    @property
    def action_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class TargetContactPlanMemory:
    context_hash: str
    trajectory_hash: str
    features: tuple[float, ...]
    action: TargetContactPlanAction
    quality_score: float

    def __post_init__(self) -> None:
        _commitment(self.context_hash, "context_hash")
        _commitment(self.trajectory_hash, "trajectory_hash")
        vector = np.asarray(self.features, dtype=np.float64)
        if (
            vector.shape != (_FEATURE_COUNT,)
            or not np.all(np.isfinite(vector))
            or not math.isfinite(self.quality_score)
        ):
            raise ValueError("target contact plan memory is invalid")


@dataclass(frozen=True)
class TargetContactPlanDecision:
    accepted: bool
    route: str
    confidence: float
    nearest_success_distance: float
    nearest_same_action_failure_distance: float | None
    selected_context_hash: str | None
    action: TargetContactPlanAction | None
    actor_hash: str


@dataclass(frozen=True)
class G1TargetContactPlanActor:
    body_hash: str
    kick_prior_hash: str
    target_contact_actor_hash: str
    source_replay_hash: str
    training_snapshot_hash: str
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    successful_memories: tuple[TargetContactPlanMemory, ...]
    failed_memories: tuple[TargetContactPlanMemory, ...]
    maximum_support_distance: float = 1.50
    failure_exclusion_margin: float = 0.05
    quality_tie_tolerance: float = 1.0e-9
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    online_hot_swap_allowed: bool = False
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        for value, label in (
            (self.body_hash, "body_hash"),
            (self.kick_prior_hash, "kick_prior_hash"),
            (self.target_contact_actor_hash, "target_contact_actor_hash"),
            (self.source_replay_hash, "source_replay_hash"),
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
            or not 0.25 <= self.maximum_support_distance <= 4.0
            or not 0.0 <= self.failure_exclusion_margin <= 0.5
            or not 0.0 <= self.quality_tie_tolerance <= 1.0e-6
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.online_hot_swap_allowed
        ):
            raise ValueError("target contact plan actor violates its SIM-only contract")

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
            "algorithm": "failure_aware_nearest_verified_mode_with_quality_tie_break",
            "decision_clock": "PRE_ROLLOUT_TASK_PLAN",
            "late_stance_rewrite_allowed": False,
            "stability_plasticity_contract": {
                "stability": "reject OOD and same-action failure-dominated plans",
                "plasticity": "rank only complete teacher-free physics outcomes",
            },
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(self, features: Sequence[float]) -> TargetContactPlanDecision:
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(vector)):
            raise ValueError("target contact plan features are invalid")
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        normalized = (vector - center) / scale

        def distance(memory: TargetContactPlanMemory) -> float:
            candidate = (np.asarray(memory.features, dtype=np.float64) - center) / scale
            return float(np.linalg.norm(candidate - normalized))

        measured = [(distance(memory), memory) for memory in self.successful_memories]
        minimum_distance = min(item[0] for item in measured)
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
        return TargetContactPlanDecision(
            accepted=accepted,
            route=(
                "VERIFIED_TARGET_CONTACT_PLAN"
                if accepted
                else (
                    "TARGET_CONTACT_FAILURE_FALLBACK"
                    if failure_distance is not None
                    else "TARGET_CONTACT_OOD_FALLBACK"
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


def save_target_contact_plan_actor(actor: G1TargetContactPlanActor, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_target_contact_plan_actor(path: Path) -> G1TargetContactPlanActor:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("target contact plan actor must be an object")
    claimed = payload.pop("actor_hash", None)
    for key in (
        "feature_names",
        "algorithm",
        "decision_clock",
        "late_stance_rewrite_allowed",
        "stability_plasticity_contract",
    ):
        payload.pop(key, None)
    payload["feature_center"] = tuple(payload["feature_center"])
    payload["feature_scale"] = tuple(payload["feature_scale"])
    for key in ("successful_memories", "failed_memories"):
        payload[key] = tuple(
            TargetContactPlanMemory(
                context_hash=item["context_hash"],
                trajectory_hash=item["trajectory_hash"],
                features=tuple(item["features"]),
                action=TargetContactPlanAction(
                    **{
                        **item["action"],
                        "target_foot_velocity_xyz_mps": tuple(
                            item["action"]["target_foot_velocity_xyz_mps"]
                        ),
                    }
                ),
                quality_score=float(item["quality_score"]),
            )
            for item in payload[key]
        )
    actor = G1TargetContactPlanActor(**payload)
    if claimed != actor.actor_hash:
        raise ValueError("target contact plan actor hash mismatch")
    return actor


__all__ = [
    "G1TargetContactPlanActor",
    "TargetContactPlanAction",
    "TargetContactPlanDecision",
    "TargetContactPlanMemory",
    "load_target_contact_plan_actor",
    "planned_contact_mode_features",
    "save_target_contact_plan_actor",
]
