"""Role-bound runtime RECEIVE control from measured pass-arrival state.

The actor selects a bounded contact geometry and phase-alignment law after a
stable incoming ball has been observed.  It never emits pose, joint, torque,
or ball commands.  The selected law is latched for stability while the causal
strike option continues to close phase error from fresh physics observations
on every control frame.
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

from rosclaw_soccer.sim.contracts import hash_json

RUNTIME_RECEIVE_FEATURE_NAMES = (
    "ball_local_x_m",
    "ball_local_y_m",
    "ball_local_vx_mps",
    "ball_local_vy_mps",
    "ball_arrival_eta_sec",
    "predicted_ball_y_at_contact_pocket_m",
    "pelvis_local_y_m",
    "joint_velocity_rms_rad_s",
    "policy_frame",
)
RUNTIME_RECEIVE_FEATURE_RESOLUTION = (
    0.04,
    0.015,
    0.25,
    0.06,
    0.15,
    0.05,
    0.02,
    0.02,
    2.0,
)
_FEATURE_COUNT = len(RUNTIME_RECEIVE_FEATURE_NAMES)
_SCHEMA = "rosclaw_soccer.g1_runtime_receive_actor.v1"


def _commitment(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{label} must be a SHA-256 commitment")
    return value


def runtime_receive_features(
    *,
    ball_local_position_m: Sequence[float],
    ball_local_velocity_mps: Sequence[float],
    ball_arrival_eta_sec: float,
    pelvis_local_position_m: Sequence[float],
    joint_velocity_rad_s: Sequence[float],
    policy_frame: int,
) -> tuple[float, ...]:
    """Build the non-privileged observation shared by training and runtime."""

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
        or isinstance(policy_frame, bool)
        or not 0 <= policy_frame <= 1_000
    ):
        raise ValueError("runtime RECEIVE features must be finite G1 observations")
    predicted_y = float(ball_position[1] + ball_velocity[1] * ball_arrival_eta_sec)
    return (
        float(ball_position[0]),
        float(ball_position[1]),
        float(ball_velocity[0]),
        float(ball_velocity[1]),
        float(ball_arrival_eta_sec),
        predicted_y,
        float(pelvis_position[1]),
        float(np.sqrt(np.mean(np.square(joint_velocity)))),
        float(policy_frame),
    )


@dataclass(frozen=True)
class RuntimeReceiveAction:
    """Bounded finisher action; phase correction remains feedback controlled."""

    maximum_arrival_advance_frames: int = 18
    arrival_alignment_tolerance_sec: float = 0.02
    stance_offset_x_m: float = -0.08
    stance_offset_y_m: float = -0.04
    contact_policy_frame: int = 252
    foot_yaw_offset_rad: float = -0.04
    foot_pitch_offset_rad: float = 0.01
    activation_ceiling: str = "SIM_ONLY"
    direct_joint_torque_output: bool = False

    def __post_init__(self) -> None:
        values = (
            self.arrival_alignment_tolerance_sec,
            self.stance_offset_x_m,
            self.stance_offset_y_m,
            self.foot_yaw_offset_rad,
            self.foot_pitch_offset_rad,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or self.maximum_arrival_advance_frames not in {0, 6, 12, 18, 24, 30}
            or not 0.02 <= self.arrival_alignment_tolerance_sec <= 0.12
            or not -0.12 <= self.stance_offset_x_m <= 0.12
            or not -0.12 <= self.stance_offset_y_m <= 0.12
            or not 238 <= self.contact_policy_frame <= 258
            or not -0.12 <= self.foot_yaw_offset_rad <= 0.12
            or not -0.08 <= self.foot_pitch_offset_rad <= 0.08
            or self.activation_ceiling != "SIM_ONLY"
            or self.direct_joint_torque_output
        ):
            raise ValueError("runtime RECEIVE action exceeds its SIM-only envelope")

    @property
    def action_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class RuntimeReceiveMemory:
    context_hash: str
    trajectory_hash: str
    features: tuple[float, ...]
    action: RuntimeReceiveAction
    quality_score: float

    def __post_init__(self) -> None:
        _commitment(self.context_hash, "context_hash")
        _commitment(self.trajectory_hash, "trajectory_hash")
        vector = np.asarray(self.features, dtype=np.float64)
        if (
            vector.shape != (_FEATURE_COUNT,)
            or not np.all(np.isfinite(vector))
            or not math.isfinite(self.quality_score)
            or not 0.0 <= self.quality_score <= 10.0
        ):
            raise ValueError("runtime RECEIVE memory is invalid")


@dataclass(frozen=True)
class RuntimeReceiveDecision:
    accepted: bool
    route: str
    confidence: float
    nearest_success_distance: float
    nearest_same_action_failure_distance: float | None
    selected_context_hash: str | None
    action: RuntimeReceiveAction | None
    actor_hash: str


@dataclass(frozen=True)
class G1RuntimeReceiveActor:
    """Failure-aware actor owned only by red.finisher's RECEIVE skill."""

    body_hash: str
    kick_prior_hash: str
    roster_hash: str
    finisher_self_model_hash: str
    source_evidence_hashes: tuple[str, ...]
    training_snapshot_hash: str
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    successful_memories: tuple[RuntimeReceiveMemory, ...]
    failed_memories: tuple[RuntimeReceiveMemory, ...]
    maximum_support_distance: float = 2.75
    failure_exclusion_margin: float = 0.05
    agent_id: str = "red.finisher"
    primary_role: str = "finisher"
    tactical_intent: str = "receive"
    owned_skill: str = "first_touch"
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
            (self.training_snapshot_hash, "training_snapshot_hash"),
        ):
            _commitment(value, label)
        if not self.source_evidence_hashes:
            raise ValueError("runtime RECEIVE actor needs evidence commitments")
        for index, value in enumerate(self.source_evidence_hashes):
            _commitment(value, f"source_evidence_hashes[{index}]")
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        success_hashes = {item.trajectory_hash for item in self.successful_memories}
        failure_hashes = {item.trajectory_hash for item in self.failed_memories}
        if (
            self.schema_version != _SCHEMA
            or center.shape != (_FEATURE_COUNT,)
            or scale.shape != (_FEATURE_COUNT,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
            or len(self.successful_memories) < 2
            or len(self.failed_memories) < 2
            or success_hashes & failure_hashes
            or not math.isfinite(self.maximum_support_distance)
            or not 0.25 <= self.maximum_support_distance <= 4.0
            or not math.isfinite(self.failure_exclusion_margin)
            or not 0.0 <= self.failure_exclusion_margin <= 0.50
            or (self.agent_id, self.primary_role, self.tactical_intent, self.owned_skill)
            != ("red.finisher", "finisher", "receive", "first_touch")
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.direct_joint_torque_output
            or self.online_hot_swap_allowed
        ):
            raise ValueError("runtime RECEIVE actor violates its role-bound SIM-only contract")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            **asdict(self),
            "feature_names": list(RUNTIME_RECEIVE_FEATURE_NAMES),
            "decision_clock": "FIFTH_CONSECUTIVE_MEASURED_INCOMING_BALL_OBSERVATION",
            "control_clock": "EVERY_20MS_CAUSAL_OPTION_FRAME_AFTER_LATCH",
            "algorithm": "failure_aware_nearest_verified_receive_control_law",
            "stability_plasticity_contract": {
                "stability": "immutable latch, OOD and failure-neighbour rejection",
                "plasticity": "new complete CPU MuJoCo trajectories create a new actor",
            },
            "authority": "RECEIVE_PHASE_AND_CONTACT_GEOMETRY_ONLY",
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(self, features: Sequence[float]) -> RuntimeReceiveDecision:
        vector = np.asarray(features, dtype=np.float64)
        if vector.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(vector)):
            raise ValueError("runtime RECEIVE decision features are invalid")
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        normalized = (vector - center) / scale

        def distance(memory: RuntimeReceiveMemory) -> float:
            candidate = (np.asarray(memory.features, dtype=np.float64) - center) / scale
            return float(np.linalg.norm(candidate - normalized))

        successes = sorted(
            ((distance(memory), memory) for memory in self.successful_memories),
            key=lambda item: (
                item[0],
                -item[1].quality_score,
                item[1].context_hash,
                item[1].trajectory_hash,
            ),
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
            "VERIFIED_RUNTIME_RECEIVE_CONTROL"
            if accepted
            else (
                "RUNTIME_RECEIVE_FAILURE_MEMORY_FALLBACK"
                if failure_distance is not None
                and success_distance + self.failure_exclusion_margin >= failure_distance
                else "RUNTIME_RECEIVE_OOD_FALLBACK"
            )
        )
        confidence = (
            max(0.0, min(1.0, 1.0 - success_distance / self.maximum_support_distance))
            if accepted
            else 0.0
        )
        return RuntimeReceiveDecision(
            accepted=accepted,
            route=route,
            confidence=confidence,
            nearest_success_distance=success_distance,
            nearest_same_action_failure_distance=failure_distance,
            selected_context_hash=selected.context_hash if accepted else None,
            action=selected.action if accepted else None,
            actor_hash=self.actor_hash,
        )


def save_runtime_receive_actor(actor: G1RuntimeReceiveActor, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_runtime_receive_actor(path: Path) -> G1RuntimeReceiveActor:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime RECEIVE actor must be an object")
    claimed = payload.pop("actor_hash", None)
    for key in (
        "feature_names",
        "decision_clock",
        "control_clock",
        "algorithm",
        "stability_plasticity_contract",
        "authority",
    ):
        payload.pop(key, None)
    payload["source_evidence_hashes"] = tuple(payload["source_evidence_hashes"])
    payload["feature_center"] = tuple(payload["feature_center"])
    payload["feature_scale"] = tuple(payload["feature_scale"])
    for key in ("successful_memories", "failed_memories"):
        payload[key] = tuple(
            RuntimeReceiveMemory(
                context_hash=item["context_hash"],
                trajectory_hash=item["trajectory_hash"],
                features=tuple(item["features"]),
                action=RuntimeReceiveAction(**item["action"]),
                quality_score=item["quality_score"],
            )
            for item in payload[key]
        )
    actor = G1RuntimeReceiveActor(**payload)
    if claimed != actor.actor_hash:
        raise ValueError("runtime RECEIVE actor hash mismatch")
    return actor


__all__ = [
    "G1RuntimeReceiveActor",
    "RUNTIME_RECEIVE_FEATURE_NAMES",
    "RUNTIME_RECEIVE_FEATURE_RESOLUTION",
    "RuntimeReceiveAction",
    "RuntimeReceiveDecision",
    "RuntimeReceiveMemory",
    "load_runtime_receive_actor",
    "runtime_receive_features",
    "save_runtime_receive_actor",
]
