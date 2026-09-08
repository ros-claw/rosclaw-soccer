"""Data-bound local experts for conservative strike coordination.

The memory routes only between already bounded S209 coordination actors.  It
cannot create a joint command, change a strike phase, or extrapolate outside
the support of verified CPU MuJoCo trajectories.  Unsupported and remembered
negative contexts explicitly abstain so the frozen parent path keeps control.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.dynamic_strike_coordination import (
    DynamicStrikeCoordinationActor,
    StrikeCoordinationAction,
    StrikeCoordinationObservation,
)
from rosclaw_soccer.sim.contracts import hash_json

STRIKE_TASK_CONTEXT_FEATURES = (
    "goal_target_y_m",
    "ball_lateral_y_m",
    "nearest_opponent_distance_m",
    "shot_line_clearance_m",
    "shot_line_progress",
)

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_EXPERT_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class StrikeTaskContext:
    """Task and opponent geometry measured from authoritative world state."""

    goal_target_y_m: float
    ball_lateral_y_m: float
    nearest_opponent_distance_m: float
    shot_line_clearance_m: float
    shot_line_progress: float

    def __post_init__(self) -> None:
        values = tuple(asdict(self).values())
        if (
            any(not isinstance(value, int | float) or not math.isfinite(value) for value in values)
            or not -2.0 <= self.goal_target_y_m <= 2.0
            or not -5.0 <= self.ball_lateral_y_m <= 5.0
            or not 0.0 <= self.nearest_opponent_distance_m <= 20.0
            or not 0.0 <= self.shot_line_clearance_m <= 10.0
            or not 0.0 <= self.shot_line_progress <= 1.0
        ):
            raise ValueError("strike task context is outside its bounded envelope")

    def vector(self) -> NDArray[np.float64]:
        return np.asarray(tuple(asdict(self).values()), dtype=np.float64)


@dataclass(frozen=True)
class ContextualStrikeExpert:
    """One verified local teacher attached to its measured context."""

    expert_id: str
    context_center: tuple[float, ...]
    actor: DynamicStrikeCoordinationActor
    source_report_hash: str
    source_trajectory_digest: str

    def __post_init__(self) -> None:
        if (
            not _EXPERT_ID.fullmatch(self.expert_id)
            or len(self.context_center) != len(STRIKE_TASK_CONTEXT_FEATURES)
            or any(not math.isfinite(value) for value in self.context_center)
            or self.actor.policy_type != "parameter"
            or self.actor.activation_ceiling != "SIM_ONLY"
            or self.actor.hardware_authorized
            or not _HASH.fullmatch(self.source_report_hash)
            or not _HASH.fullmatch(self.source_trajectory_digest)
        ):
            raise ValueError("contextual strike expert is not a verified SIM-only teacher")
        StrikeTaskContext(*self.context_center)

    @property
    def expert_hash(self) -> str:
        return hash_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        actor = asdict(self.actor)
        for name in ("weights", "biases", "feature_mean", "feature_std"):
            actor[name] = list(actor[name])
        return {
            "expert_id": self.expert_id,
            "context_center": list(self.context_center),
            "actor": actor,
            "actor_hash": self.actor.actor_hash,
            "source_report_hash": self.source_report_hash,
            "source_trajectory_digest": self.source_trajectory_digest,
        }


@dataclass(frozen=True)
class ContextualStrikeNegative:
    """An unresolved failure boundary that blocks unsafe extrapolation."""

    boundary_id: str
    context_center: tuple[float, ...]
    source_report_hash: str
    source_trajectory_digest: str

    def __post_init__(self) -> None:
        if (
            not _EXPERT_ID.fullmatch(self.boundary_id)
            or len(self.context_center) != len(STRIKE_TASK_CONTEXT_FEATURES)
            or any(not math.isfinite(value) for value in self.context_center)
            or not _HASH.fullmatch(self.source_report_hash)
            or not _HASH.fullmatch(self.source_trajectory_digest)
        ):
            raise ValueError("contextual strike negative boundary is invalid")
        StrikeTaskContext(*self.context_center)

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "context_center": list(self.context_center),
            "source_report_hash": self.source_report_hash,
            "source_trajectory_digest": self.source_trajectory_digest,
        }


@dataclass(frozen=True)
class ContextualStrikeSelection:
    """Auditable routing result; abstention carries no motion authority."""

    action: StrikeCoordinationAction | None
    expert_index: int
    normalized_distance: float
    reason: str

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.normalized_distance)
            or self.normalized_distance < 0.0
            or self.reason not in {"verified-expert", "negative-memory", "out-of-support"}
            or (self.reason == "verified-expert") != (self.action is not None)
            or (self.action is None and self.expert_index != -1)
            or (self.action is not None and self.expert_index < 0)
        ):
            raise ValueError("contextual strike selection is inconsistent")


@dataclass(frozen=True)
class ContextualStrikeExpertMemory:
    """Nearest verified expert with explicit support and failure boundaries."""

    experts: tuple[ContextualStrikeExpert, ...]
    negative_boundaries: tuple[ContextualStrikeNegative, ...]
    context_scale: tuple[float, ...]
    acceptance_radius: float
    negative_radius: float
    dataset_snapshot_hash: str
    policy_type: str = "contextual_nearest_verified_expert"
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.contextual_strike_expert_memory.v1"

    def __post_init__(self) -> None:
        ids = [expert.expert_id for expert in self.experts]
        negative_ids = [boundary.boundary_id for boundary in self.negative_boundaries]
        if (
            len(self.experts) < 3
            or len(set(ids)) != len(ids)
            or len(set(negative_ids)) != len(negative_ids)
            or set(ids) & set(negative_ids)
            or len(self.context_scale) != len(STRIKE_TASK_CONTEXT_FEATURES)
            or any(not math.isfinite(value) or value <= 0.0 for value in self.context_scale)
            or not 0.05 <= self.acceptance_radius <= 2.0
            or not 0.05 <= self.negative_radius <= 2.0
            or not _HASH.fullmatch(self.dataset_snapshot_hash)
            or self.dataset_snapshot_hash
            != _memory_snapshot_hash(self.experts, self.negative_boundaries)
            or self.policy_type != "contextual_nearest_verified_expert"
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("contextual strike memory violates its SIM-only contract")

    @classmethod
    def build(
        cls,
        *,
        experts: tuple[ContextualStrikeExpert, ...],
        negative_boundaries: tuple[ContextualStrikeNegative, ...] = (),
        context_scale: tuple[float, ...] = (0.05, 0.04, 0.25, 0.08, 0.05),
        acceptance_radius: float = 0.75,
        negative_radius: float = 0.75,
    ) -> ContextualStrikeExpertMemory:
        return cls(
            experts=experts,
            negative_boundaries=negative_boundaries,
            context_scale=context_scale,
            acceptance_radius=acceptance_radius,
            negative_radius=negative_radius,
            dataset_snapshot_hash=_memory_snapshot_hash(experts, negative_boundaries),
        )

    @property
    def memory_hash(self) -> str:
        return hash_json(self.to_dict())

    def select(
        self,
        *,
        context: StrikeTaskContext,
        observation: StrikeCoordinationObservation,
    ) -> ContextualStrikeSelection:
        vector = context.vector()
        scale = np.asarray(self.context_scale, dtype=np.float64)
        expert_distances = np.asarray(
            [
                np.linalg.norm(
                    (vector - np.asarray(expert.context_center, dtype=np.float64)) / scale
                )
                for expert in self.experts
            ],
            dtype=np.float64,
        )
        expert_index = int(np.argmin(expert_distances))
        expert_distance = float(expert_distances[expert_index])
        if self.negative_boundaries:
            negative_distance = float(
                min(
                    np.linalg.norm(
                        (vector - np.asarray(boundary.context_center, dtype=np.float64)) / scale
                    )
                    for boundary in self.negative_boundaries
                )
            )
            if negative_distance <= self.negative_radius and negative_distance <= expert_distance:
                return ContextualStrikeSelection(None, -1, negative_distance, "negative-memory")
        if expert_distance > self.acceptance_radius:
            return ContextualStrikeSelection(None, -1, expert_distance, "out-of-support")
        return ContextualStrikeSelection(
            self.experts[expert_index].actor.act(observation),
            expert_index,
            expert_distance,
            "verified-expert",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experts": [expert.to_dict() for expert in self.experts],
            "negative_boundaries": [boundary.to_dict() for boundary in self.negative_boundaries],
            "context_scale": list(self.context_scale),
            "acceptance_radius": self.acceptance_radius,
            "negative_radius": self.negative_radius,
            "dataset_snapshot_hash": self.dataset_snapshot_hash,
            "policy_type": self.policy_type,
            "activation_ceiling": self.activation_ceiling,
            "hardware_authorized": self.hardware_authorized,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_mapping(cls, raw: object) -> ContextualStrikeExpertMemory:
        if not isinstance(raw, Mapping):
            raise ValueError("contextual strike memory payload must be an object")
        values = dict(raw)
        raw_experts = values.get("experts")
        raw_negatives = values.get("negative_boundaries")
        if not isinstance(raw_experts, list) or not isinstance(raw_negatives, list):
            raise ValueError("contextual strike memory entries must be arrays")
        experts: list[ContextualStrikeExpert] = []
        for raw_expert in raw_experts:
            if not isinstance(raw_expert, Mapping):
                raise ValueError("contextual strike expert payload is invalid")
            expert_values = dict(raw_expert)
            actor_hash = expert_values.pop("actor_hash", None)
            actor = _actor_from_mapping(expert_values.pop("actor", None))
            if actor_hash != actor.actor_hash:
                raise ValueError("contextual strike expert actor hash changed")
            expert_values["context_center"] = tuple(expert_values["context_center"])
            experts.append(ContextualStrikeExpert(actor=actor, **expert_values))
        negatives: list[ContextualStrikeNegative] = []
        for raw_negative in raw_negatives:
            if not isinstance(raw_negative, Mapping):
                raise ValueError("contextual strike negative payload is invalid")
            negative_values = dict(raw_negative)
            negative_values["context_center"] = tuple(negative_values["context_center"])
            negatives.append(ContextualStrikeNegative(**negative_values))
        values["experts"] = tuple(experts)
        values["negative_boundaries"] = tuple(negatives)
        values["context_scale"] = tuple(values["context_scale"])
        return cls(**values)


def build_strike_task_context(
    *,
    goal_target_m: tuple[float, float],
    ball_position_m: NDArray[np.float64],
    opponent_positions_m: NDArray[np.float64],
) -> StrikeTaskContext:
    """Measure target and nearest shot-line opponent without mutating state."""

    ball = np.asarray(ball_position_m, dtype=np.float64)
    opponents = np.asarray(opponent_positions_m, dtype=np.float64)
    target = np.asarray(goal_target_m, dtype=np.float64)
    if (
        ball.shape != (2,)
        or target.shape != (2,)
        or opponents.ndim != 2
        or opponents.shape[1:] != (2,)
        or len(opponents) == 0
        or not np.all(np.isfinite(ball))
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(opponents))
    ):
        raise ValueError("strike task context requires finite 2-D world geometry")
    shot_line = target - ball
    denominator = float(shot_line @ shot_line)
    if denominator <= 1.0e-9:
        raise ValueError("strike target and ball are degenerate")
    candidates: list[tuple[float, float, float]] = []
    for opponent in opponents:
        relative = opponent - ball
        progress = float(np.clip((relative @ shot_line) / denominator, 0.0, 1.0))
        closest = ball + progress * shot_line
        candidates.append(
            (
                float(np.linalg.norm(opponent - closest)),
                float(np.linalg.norm(relative)),
                progress,
            )
        )
    clearance, distance, progress = min(candidates)
    return StrikeTaskContext(
        goal_target_y_m=float(target[1]),
        ball_lateral_y_m=float(ball[1]),
        nearest_opponent_distance_m=distance,
        shot_line_clearance_m=clearance,
        shot_line_progress=progress,
    )


def _memory_snapshot_hash(
    experts: tuple[ContextualStrikeExpert, ...],
    negative_boundaries: tuple[ContextualStrikeNegative, ...],
) -> str:
    return hash_json(
        {
            "experts": [expert.to_dict() for expert in experts],
            "negative_boundaries": [boundary.to_dict() for boundary in negative_boundaries],
        }
    )


def _actor_from_mapping(raw: object) -> DynamicStrikeCoordinationActor:
    if not isinstance(raw, Mapping):
        raise ValueError("contextual strike expert actor payload is invalid")
    values = dict(raw)
    for name in ("weights", "biases", "feature_mean", "feature_std"):
        vector = values.get(name)
        if not isinstance(vector, list):
            raise ValueError("contextual strike actor vectors must be arrays")
        values[name] = tuple(float(item) for item in vector)
    return DynamicStrikeCoordinationActor(**values)


__all__ = [
    "ContextualStrikeExpert",
    "ContextualStrikeExpertMemory",
    "ContextualStrikeNegative",
    "ContextualStrikeSelection",
    "STRIKE_TASK_CONTEXT_FEATURES",
    "StrikeTaskContext",
    "build_strike_task_context",
]
