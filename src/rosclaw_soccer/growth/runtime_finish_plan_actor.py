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
from dataclasses import asdict, dataclass, replace
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
RUNTIME_FINISH_PLAN_ACTION_NAMES = (
    "maximum_arrival_advance_frames",
    "arrival_alignment_tolerance_sec",
    "stance_offset_x_m",
    "stance_offset_y_m",
    "contact_policy_frame",
    "foot_yaw_offset_rad",
    "foot_pitch_offset_rad",
    "target_foot_velocity_x_mps",
    "target_foot_velocity_y_mps",
    "target_foot_velocity_z_mps",
)
RUNTIME_FINISH_PLAN_CRITIC_NAMES = (
    "safe_probability",
    "intended_foot_probability",
    "clear_outcome_probability",
    "strict_success_probability",
    "precision_value",
    "post_contact_stability_value",
)
_ACTION_COUNT = len(RUNTIME_FINISH_PLAN_ACTION_NAMES)
_CRITIC_COUNT = len(RUNTIME_FINISH_PLAN_CRITIC_NAMES)
_CRITIC_INPUT_COUNT = _FEATURE_COUNT + _ACTION_COUNT
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
class RuntimeFinishPlanCriticHead:
    """One context-held-out kernel critic with no executable payload."""

    training_context_hashes: tuple[str, ...]
    normalized_inputs: tuple[tuple[float, ...], ...]
    coefficients: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        inputs = np.asarray(self.normalized_inputs, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if (
            len(self.training_context_hashes) < 4
            or len(set(self.training_context_hashes)) != len(self.training_context_hashes)
            or any(
                _commitment(value, "training_context_hash") != value
                for value in self.training_context_hashes
            )
            or inputs.ndim != 2
            or inputs.shape[0] < 32
            or inputs.shape[1] != _CRITIC_INPUT_COUNT
            or coefficients.shape != (inputs.shape[0], _CRITIC_COUNT)
            or not np.all(np.isfinite(inputs))
            or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError("runtime finish plan critic head is invalid")


@dataclass(frozen=True)
class RuntimeFinishPlanContinuousPolicy:
    """Bounded continuous proposal layer guarded by an immutable parent actor."""

    parent_actor_hash: str
    parent_training_snapshot_hash: str
    critic_training_snapshot_hash: str
    input_center: tuple[float, ...]
    input_scale: tuple[float, ...]
    critic_heads: tuple[RuntimeFinishPlanCriticHead, ...]
    kernel_bandwidth: float = 4.0
    ridge_penalty: float = 0.1
    nearest_success_count: int = 5
    interpolation_bandwidth: float = 1.0
    interpolation_alphas: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0)
    minimum_safety_floor: float = 0.65
    maximum_ensemble_spread: float = 0.75
    minimum_strict_advantage: float = 0.02
    minimum_precision_advantage: float = 0.015
    maximum_stability_regression: float = 0.12
    maximum_intended_foot_regression: float = 0.05
    activation_ceiling: str = "SIM_ONLY"
    direct_joint_torque_output: bool = False
    feedback_evidence_hashes: tuple[str, ...] = ()
    failed_continuous_inputs: tuple[tuple[float, ...], ...] = ()
    failure_exclusion_distance: float = 0.25

    def __post_init__(self) -> None:
        for value, label in (
            (self.parent_actor_hash, "parent_actor_hash"),
            (self.parent_training_snapshot_hash, "parent_training_snapshot_hash"),
            (self.critic_training_snapshot_hash, "critic_training_snapshot_hash"),
        ):
            _commitment(value, label)
        center = np.asarray(self.input_center, dtype=np.float64)
        scale = np.asarray(self.input_scale, dtype=np.float64)
        alphas = np.asarray(self.interpolation_alphas, dtype=np.float64)
        failures = np.asarray(self.failed_continuous_inputs, dtype=np.float64)
        if (
            center.shape != (_CRITIC_INPUT_COUNT,)
            or scale.shape != (_CRITIC_INPUT_COUNT,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
            or len(self.critic_heads) != 4
            or not math.isfinite(self.kernel_bandwidth)
            or not 0.5 <= self.kernel_bandwidth <= 8.0
            or not math.isfinite(self.ridge_penalty)
            or not 1.0e-4 <= self.ridge_penalty <= 10.0
            or not 2 <= self.nearest_success_count <= 8
            or not math.isfinite(self.interpolation_bandwidth)
            or not 0.25 <= self.interpolation_bandwidth <= 4.0
            or alphas.ndim != 1
            or not 1 <= alphas.size <= 8
            or not np.all(np.isfinite(alphas))
            or np.any(alphas <= 0.0)
            or np.any(alphas > 1.0)
            or len(set(self.interpolation_alphas)) != len(self.interpolation_alphas)
            or not 0.5 <= self.minimum_safety_floor <= 1.0
            or not 0.0 <= self.maximum_ensemble_spread <= 1.0
            or not 0.0 <= self.minimum_strict_advantage <= 0.5
            or not 0.0 <= self.minimum_precision_advantage <= 0.5
            or not 0.0 <= self.maximum_stability_regression <= 0.5
            or not 0.0 <= self.maximum_intended_foot_regression <= 0.5
            or self.activation_ceiling != "SIM_ONLY"
            or self.direct_joint_torque_output
            or any(
                _commitment(value, "feedback_evidence_hash") != value
                for value in self.feedback_evidence_hashes
            )
            or (
                bool(self.failed_continuous_inputs)
                and (
                    failures.ndim != 2
                    or failures.shape[1] != _CRITIC_INPUT_COUNT
                    or not np.all(np.isfinite(failures))
                    or not self.feedback_evidence_hashes
                )
            )
            or not 0.05 <= self.failure_exclusion_distance <= 1.0
        ):
            raise ValueError("runtime finish plan continuous policy is invalid")

    def predict(self, features: object, action: RuntimeFinishPlanAction) -> np.ndarray:
        """Return four conservative multi-task predictions in ``[0, 1]``."""

        feature_vector = np.asarray(features, dtype=np.float64)
        if feature_vector.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(feature_vector)):
            raise ValueError("runtime finish plan critic features are invalid")
        raw = np.concatenate((feature_vector, runtime_finish_plan_action_vector(action)))
        normalized = (raw - np.asarray(self.input_center)) / np.asarray(self.input_scale)
        predictions: list[np.ndarray] = []
        for head in self.critic_heads:
            inputs = np.asarray(head.normalized_inputs, dtype=np.float64)
            distances = np.sum(np.square(inputs - normalized), axis=1)
            kernel = np.exp(-distances / (2.0 * self.kernel_bandwidth**2))
            predictions.append(kernel @ np.asarray(head.coefficients, dtype=np.float64))
        return np.clip(np.asarray(predictions, dtype=np.float64), 0.0, 1.0)


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
    used_continuous_policy: bool = False
    parent_action_hash: str | None = None
    critic_safe_floor: float | None = None
    critic_strict_mean: float | None = None
    critic_precision_mean: float | None = None
    critic_stability_floor: float | None = None
    critic_maximum_spread: float | None = None


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
    continuous_policy: RuntimeFinishPlanContinuousPolicy | None = None
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
        if self.continuous_policy is not None:
            parent = replace(
                self,
                training_snapshot_hash=self.continuous_policy.parent_training_snapshot_hash,
                continuous_policy=None,
            )
            if parent.actor_hash != self.continuous_policy.parent_actor_hash:
                raise ValueError("runtime finish plan continuous parent binding changed")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        serialized = asdict(self)
        if self.continuous_policy is None:
            # Preserve byte-for-byte semantic hashing of all v1 parent actors.
            serialized.pop("continuous_policy")
        elif not self.continuous_policy.feedback_evidence_hashes:
            continuous = serialized["continuous_policy"]
            assert isinstance(continuous, dict)
            continuous.pop("feedback_evidence_hashes")
            continuous.pop("failed_continuous_inputs")
            continuous.pop("failure_exclusion_distance")
        value: dict[str, Any] = {
            **serialized,
            "feature_names": list(PREPARED_FINISH_PLAN_FEATURE_NAMES),
            "action_names": list(RUNTIME_FINISH_PLAN_ACTION_NAMES),
            "critic_names": list(RUNTIME_FINISH_PLAN_CRITIC_NAMES),
            "algorithm": (
                "failure_aware_nearest_verified_joint_finish_plan"
                if self.continuous_policy is None
                else "parent_anchored_continuous_joint_plan_with_four_head_kernel_critic"
            ),
            "decision_clock": "PRE_ROLLOUT_SHARED_TEAM_INTENT",
            "authority": "RECEIVE_GEOMETRY_PHASE_AND_TASK_SPACE_TARGET_ONLY",
            "joint_torque_owner": "content_bound_neural_contact_actor",
            "stability_plasticity_contract": {
                "stability": "immutable latch, OOD and same-plan failure exclusion",
                "plasticity": "new complete CPU MuJoCo outcomes create a new artifact",
            },
        }
        if self.continuous_policy is None:
            # These keys did not exist in the original v1 serialization.
            value.pop("action_names")
            value.pop("critic_names")
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
        decision = RuntimeFinishPlanDecision(
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
        if not accepted or self.continuous_policy is None or success_distance <= 1.0e-12:
            return decision
        return self._continuous_decision(vector, selected, decision)

    def _continuous_decision(
        self,
        features: np.ndarray,
        parent_memory: RuntimeFinishPlanMemory,
        parent_decision: RuntimeFinishPlanDecision,
    ) -> RuntimeFinishPlanDecision:
        policy = self.continuous_policy
        assert policy is not None
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        ranked = sorted(
            (
                (
                    float(
                        np.linalg.norm(
                            (np.asarray(memory.features, dtype=np.float64) - features) / scale
                        )
                    ),
                    memory,
                )
                for memory in self.successful_memories
            ),
            key=lambda item: (
                item[0],
                -item[1].quality_score,
                item[1].action.action_hash,
                item[1].trajectory_hash,
            ),
        )[: policy.nearest_success_count]
        weights = np.asarray(
            [
                math.exp(-(distance**2) / (2.0 * policy.interpolation_bandwidth**2))
                for distance, _ in ranked
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 1.0e-12:
            return parent_decision
        centroid = np.average(
            np.asarray(
                [runtime_finish_plan_action_vector(memory.action) for _, memory in ranked],
                dtype=np.float64,
            ),
            axis=0,
            weights=weights,
        )
        parent_action = parent_memory.action
        parent_vector = runtime_finish_plan_action_vector(parent_action)
        parent_prediction = policy.predict(features, parent_action)
        parent_mean = np.mean(parent_prediction, axis=0)
        eligible: list[
            tuple[tuple[float, float, float, float], RuntimeFinishPlanAction, np.ndarray]
        ] = []
        seen = {parent_action.action_hash}
        for alpha in policy.interpolation_alphas:
            proposal_vector = (1.0 - alpha) * parent_vector + alpha * centroid
            # Arrival advance is intentionally inherited from the verified parent;
            # the causal feedback option, rather than this pre-rollout actor, owns it.
            proposal_vector[0] = parent_vector[0]
            proposal = runtime_finish_plan_action_from_vector(proposal_vector)
            if proposal.action_hash in seen:
                continue
            seen.add(proposal.action_hash)
            if policy.failed_continuous_inputs:
                proposal_input = np.concatenate(
                    (features, runtime_finish_plan_action_vector(proposal))
                )
                normalized_input = (proposal_input - np.asarray(policy.input_center)) / np.asarray(
                    policy.input_scale
                )
                failed_inputs = np.asarray(policy.failed_continuous_inputs, dtype=np.float64)
                if (
                    float(np.min(np.linalg.norm(failed_inputs - normalized_input, axis=1)))
                    <= policy.failure_exclusion_distance
                ):
                    continue
            prediction = policy.predict(features, proposal)
            means = np.mean(prediction, axis=0)
            floors = np.min(prediction, axis=0)
            spread = float(np.max(np.ptp(prediction, axis=0)))
            if (
                floors[0] < policy.minimum_safety_floor
                or floors[0] + 0.02 < float(np.min(parent_prediction[:, 0]))
                or means[1] + policy.maximum_intended_foot_regression < parent_mean[1]
                or means[3] < parent_mean[3] + policy.minimum_strict_advantage
                or means[4] < parent_mean[4] + policy.minimum_precision_advantage
                or floors[5] + policy.maximum_stability_regression
                < float(np.min(parent_prediction[:, 5]))
                or spread > policy.maximum_ensemble_spread
            ):
                continue
            utility = float(means[3] + means[4] + 0.15 * means[1] + 0.10 * means[5])
            eligible.append(((utility, means[4], means[3], -spread), proposal, prediction))
        if not eligible:
            return replace(parent_decision, parent_action_hash=parent_action.action_hash)
        _, action, prediction = max(eligible, key=lambda item: item[0])
        means = np.mean(prediction, axis=0)
        floors = np.min(prediction, axis=0)
        return replace(
            parent_decision,
            route="VERIFIED_RUNTIME_CONTINUOUS_FINISH_PLAN",
            confidence=min(parent_decision.confidence, float(floors[0])),
            action=action,
            used_continuous_policy=True,
            parent_action_hash=parent_action.action_hash,
            critic_safe_floor=float(floors[0]),
            critic_strict_mean=float(means[3]),
            critic_precision_mean=float(means[4]),
            critic_stability_floor=float(floors[5]),
            critic_maximum_spread=float(np.max(np.ptp(prediction, axis=0))),
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
        "action_names",
        "critic_names",
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
    continuous = payload.get("continuous_policy")
    if isinstance(continuous, dict):
        payload["continuous_policy"] = _continuous_policy_from_dict(continuous)
    for key in ("successful_memories", "failed_memories"):
        payload[key] = tuple(_memory_from_dict(item) for item in payload[key])
    actor = G1RuntimeFinishPlanActor(**payload)
    if claimed != actor.actor_hash:
        raise ValueError("runtime finish plan actor hash mismatch")
    return actor


def runtime_finish_plan_action_vector(action: RuntimeFinishPlanAction) -> np.ndarray:
    """Encode the bounded high-level action for training and critic inference."""

    receive = action.receive
    return np.asarray(
        (
            receive.maximum_arrival_advance_frames,
            receive.arrival_alignment_tolerance_sec,
            receive.stance_offset_x_m,
            receive.stance_offset_y_m,
            receive.contact_policy_frame,
            receive.foot_yaw_offset_rad,
            receive.foot_pitch_offset_rad,
            *action.target.target_foot_velocity_xyz_mps,
        ),
        dtype=np.float64,
    )


def runtime_finish_plan_action_from_vector(values: Sequence[float]) -> RuntimeFinishPlanAction:
    """Decode a convex proposal while re-applying every action envelope."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (_ACTION_COUNT,) or not np.all(np.isfinite(vector)):
        raise ValueError("runtime finish plan action vector is invalid")
    advances = np.asarray((0, 6, 12, 18, 24, 30), dtype=np.int64)
    advance = int(advances[int(np.argmin(np.abs(advances - vector[0])))])
    receive = RuntimeReceiveAction(
        maximum_arrival_advance_frames=advance,
        arrival_alignment_tolerance_sec=float(np.clip(vector[1], 0.02, 0.12)),
        stance_offset_x_m=float(np.clip(vector[2], -0.12, 0.12)),
        stance_offset_y_m=float(np.clip(vector[3], -0.12, 0.12)),
        contact_policy_frame=int(np.clip(round(float(vector[4])), 238, 258)),
        foot_yaw_offset_rad=float(np.clip(vector[5], -0.12, 0.12)),
        foot_pitch_offset_rad=float(np.clip(vector[6], -0.08, 0.08)),
    )
    target_values = np.clip(
        vector[7:10],
        np.asarray((5.0, -6.0, -3.0)),
        np.asarray((12.0, 6.0, 6.0)),
    )
    target = RuntimeContactTargetAction(
        (float(target_values[0]), float(target_values[1]), float(target_values[2]))
    )
    return RuntimeFinishPlanAction(receive=receive, target=target)


def _continuous_policy_from_dict(value: dict[str, Any]) -> RuntimeFinishPlanContinuousPolicy:
    payload = dict(value)
    payload["input_center"] = tuple(payload["input_center"])
    payload["input_scale"] = tuple(payload["input_scale"])
    payload["interpolation_alphas"] = tuple(payload["interpolation_alphas"])
    payload["feedback_evidence_hashes"] = tuple(payload.get("feedback_evidence_hashes", ()))
    payload["failed_continuous_inputs"] = tuple(
        tuple(row) for row in payload.get("failed_continuous_inputs", ())
    )
    payload["critic_heads"] = tuple(
        RuntimeFinishPlanCriticHead(
            training_context_hashes=tuple(head["training_context_hashes"]),
            normalized_inputs=tuple(tuple(row) for row in head["normalized_inputs"]),
            coefficients=tuple(tuple(row) for row in head["coefficients"]),
        )
        for head in payload["critic_heads"]
    )
    return RuntimeFinishPlanContinuousPolicy(**payload)


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
    "RuntimeFinishPlanContinuousPolicy",
    "RuntimeFinishPlanCriticHead",
    "RuntimeFinishPlanDecision",
    "RuntimeFinishPlanMemory",
    "RUNTIME_FINISH_PLAN_ACTION_NAMES",
    "RUNTIME_FINISH_PLAN_CRITIC_NAMES",
    "load_runtime_finish_plan_actor",
    "prepared_finish_plan_features",
    "runtime_finish_plan_action_from_vector",
    "runtime_finish_plan_action_vector",
    "save_runtime_finish_plan_actor",
]
