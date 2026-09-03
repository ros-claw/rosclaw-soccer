"""Failure-aware joint RECEIVE-and-strike planning from shared team intent.

The actor resolves a credit-assignment error discovered in S143--S149: contact
geometry, phase alignment and task-space foot velocity are coupled at impact.
It selects those bounded high-level values together while leaving every joint
torque to the content-bound neural contact actor in the simulator runtime.
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

from rosclaw_soccer.growth.runtime_contact_target_actor import RuntimeContactTargetAction
from rosclaw_soccer.growth.runtime_receive_actor import RuntimeReceiveAction
from rosclaw_soccer.sim.contracts import hash_json

PREPARED_FINISH_PLAN_FEATURE_NAMES = (
    "receiver_lane_m",
    "reception_target_x_m",
    "passer_ball_local_x_m",
    "passer_ball_local_y_m",
    "ball_ground_friction",
    "passer_yaw_rad",
    "passer_stance_offset_x_m",
    "passer_stance_offset_y_m",
    "passer_swing_speed_scale",
)
_FEATURE_COUNT = len(PREPARED_FINISH_PLAN_FEATURE_NAMES)
_SCHEMA = "rosclaw_soccer.g1_runtime_finish_plan_actor.v1"


def prepared_finish_plan_features(
    *,
    receiver_lane_m: float,
    reception_target_x_m: float,
    passer_ball_local_xy_m: Sequence[float],
    ball_ground_friction: float,
    passer_yaw_rad: float,
    passer_stance_offset_xy_m: Sequence[float],
    passer_swing_speed_scale: float,
) -> tuple[float, ...]:
    """Build the pre-rollout role-plan observation shared by train and runtime."""

    ball = np.asarray(passer_ball_local_xy_m, dtype=np.float64)
    stance = np.asarray(passer_stance_offset_xy_m, dtype=np.float64)
    scalars = np.asarray(
        (
            receiver_lane_m,
            reception_target_x_m,
            ball_ground_friction,
            passer_yaw_rad,
            passer_swing_speed_scale,
        ),
        dtype=np.float64,
    )
    if (
        ball.shape != (2,)
        or stance.shape != (2,)
        or not np.all(np.isfinite(ball))
        or not np.all(np.isfinite(stance))
        or not np.all(np.isfinite(scalars))
    ):
        raise ValueError("prepared finish plan features must be finite team intent")
    return (
        float(receiver_lane_m),
        float(reception_target_x_m),
        float(ball[0]),
        float(ball[1]),
        float(ball_ground_friction),
        float(passer_yaw_rad),
        float(stance[0]),
        float(stance[1]),
        float(passer_swing_speed_scale),
    )


def _commitment(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{label} must be a SHA-256 commitment")
    return value


@dataclass(frozen=True)
class RuntimeFinishPlanAction:
    """Bounded high-level plan with no pose, joint or torque authority."""

    receive: RuntimeReceiveAction
    target: RuntimeContactTargetAction
    activation_ceiling: str = "SIM_ONLY"
    direct_joint_torque_output: bool = False

    def __post_init__(self) -> None:
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.direct_joint_torque_output
            or self.receive.activation_ceiling != "SIM_ONLY"
            or self.receive.direct_joint_torque_output
            or self.target.activation_ceiling != "SIM_ONLY"
            or self.target.direct_joint_torque_output
        ):
            raise ValueError("runtime finish plan exceeds its high-level SIM-only envelope")

    @property
    def action_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class RuntimeFinishPlanMemory:
    context_hash: str
    trajectory_hash: str
    features: tuple[float, ...]
    action: RuntimeFinishPlanAction
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
            raise ValueError("runtime finish plan memory is invalid")


@dataclass(frozen=True)
class RuntimeFinishPlanDecision:
    accepted: bool
    route: str
    confidence: float
    nearest_success_distance: float
    nearest_same_action_failure_distance: float | None
    selected_context_hash: str | None
    action: RuntimeFinishPlanAction | None
    actor_hash: str


@dataclass(frozen=True)
class G1RuntimeFinishPlanActor:
    """Role-bound joint planner for ``red.finisher.receive_and_strike``."""

    body_hash: str
    kick_prior_hash: str
    roster_hash: str
    finisher_self_model_hash: str
    neural_contact_actor_hash: str
    contact_handoff_actor_hash: str
    contact_handoff_offset_frames: int
    source_evidence_hashes: tuple[str, ...]
    training_snapshot_hash: str
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    successful_memories: tuple[RuntimeFinishPlanMemory, ...]
    failed_memories: tuple[RuntimeFinishPlanMemory, ...]
    maximum_support_distance: float = 2.75
    failure_exclusion_margin: float = 0.05
    agent_id: str = "red.finisher"
    owned_skill: str = "receive_and_strike_plan"
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
            (self.contact_handoff_actor_hash, "contact_handoff_actor_hash"),
            (self.training_snapshot_hash, "training_snapshot_hash"),
        ):
            _commitment(value, label)
        for index, value in enumerate(self.source_evidence_hashes):
            _commitment(value, f"source_evidence_hashes[{index}]")
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        success_hashes = {memory.trajectory_hash for memory in self.successful_memories}
        failure_hashes = {memory.trajectory_hash for memory in self.failed_memories}
        if (
            self.schema_version != _SCHEMA
            or not self.source_evidence_hashes
            or center.shape != (_FEATURE_COUNT,)
            or scale.shape != (_FEATURE_COUNT,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
            or len(self.successful_memories) < 4
            or len(self.failed_memories) < 8
            or success_hashes & failure_hashes
            or not 0 <= self.contact_handoff_offset_frames <= 30
            or not 0.25 <= self.maximum_support_distance <= 4.0
            or not 0.0 <= self.failure_exclusion_margin <= 0.50
            or (self.agent_id, self.owned_skill) != ("red.finisher", "receive_and_strike_plan")
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.direct_joint_torque_output
            or self.online_hot_swap_allowed
        ):
            raise ValueError("runtime finish plan actor violates its role-bound contract")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "feature_names": list(PREPARED_FINISH_PLAN_FEATURE_NAMES),
            "algorithm": "failure_aware_nearest_verified_joint_finish_plan",
            "decision_clock": "PRE_ROLLOUT_SHARED_TEAM_INTENT",
            "authority": "RECEIVE_GEOMETRY_PHASE_AND_TASK_SPACE_TARGET_ONLY",
            "joint_torque_owner": "content_bound_neural_contact_actor",
            "stability_plasticity_contract": {
                "stability": "immutable latch, OOD and same-plan failure exclusion",
                "plasticity": "new complete CPU MuJoCo outcomes create a new artifact",
            },
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(self, features: Sequence[float]) -> RuntimeFinishPlanDecision:
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(vector)):
            raise ValueError("runtime finish plan features are invalid")
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        normalized = (vector - center) / scale

        def distance(memory: RuntimeFinishPlanMemory) -> float:
            candidate = (np.asarray(memory.features, dtype=np.float64) - center) / scale
            return float(np.linalg.norm(candidate - normalized))

        successes = sorted(
            ((distance(memory), memory) for memory in self.successful_memories),
            key=lambda item: (
                item[0],
                -item[1].quality_score,
                item[1].action.action_hash,
                item[1].trajectory_hash,
            ),
        )
        success_distance, selected = successes[0]
        failures = sorted(
            distance(memory) for memory in self.failed_memories if memory.action == selected.action
        )
        failure_distance = failures[0] if failures else None
        accepted = bool(
            success_distance <= self.maximum_support_distance
            and (
                failure_distance is None
                or success_distance + self.failure_exclusion_margin < failure_distance
            )
        )
        route = (
            "VERIFIED_RUNTIME_FINISH_PLAN"
            if accepted
            else (
                "RUNTIME_FINISH_PLAN_OOD_FALLBACK"
                if success_distance > self.maximum_support_distance
                else "RUNTIME_FINISH_PLAN_FAILURE_MEMORY_FALLBACK"
            )
        )
        return RuntimeFinishPlanDecision(
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


def save_runtime_finish_plan_actor(actor: G1RuntimeFinishPlanActor, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_runtime_finish_plan_actor(path: Path) -> G1RuntimeFinishPlanActor:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime finish plan actor must be an object")
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
    for key in ("successful_memories", "failed_memories"):
        payload[key] = tuple(_memory_from_dict(item) for item in payload[key])
    actor = G1RuntimeFinishPlanActor(**payload)
    if claimed != actor.actor_hash:
        raise ValueError("runtime finish plan actor hash mismatch")
    return actor


def _memory_from_dict(value: dict[str, Any]) -> RuntimeFinishPlanMemory:
    action = value["action"]
    return RuntimeFinishPlanMemory(
        context_hash=value["context_hash"],
        trajectory_hash=value["trajectory_hash"],
        features=tuple(value["features"]),
        action=RuntimeFinishPlanAction(
            receive=RuntimeReceiveAction(**action["receive"]),
            target=RuntimeContactTargetAction(
                target_foot_velocity_xyz_mps=tuple(
                    action["target"]["target_foot_velocity_xyz_mps"]
                ),
                activation_ceiling=action["target"]["activation_ceiling"],
                direct_joint_torque_output=action["target"]["direct_joint_torque_output"],
            ),
            activation_ceiling=action["activation_ceiling"],
            direct_joint_torque_output=action["direct_joint_torque_output"],
        ),
        quality_score=float(value["quality_score"]),
    )


__all__ = [
    "G1RuntimeFinishPlanActor",
    "RuntimeFinishPlanAction",
    "RuntimeFinishPlanDecision",
    "RuntimeFinishPlanMemory",
    "load_runtime_finish_plan_actor",
    "prepared_finish_plan_features",
    "save_runtime_finish_plan_actor",
]
