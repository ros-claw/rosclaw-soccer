"""Content-bound reactive route actor for SIM-only multi-agent movement.

The actor predicts a planar world-frame locomotion command from proprioception
and the current football context.  It never emits pose, joint, torque, ROS or
hardware commands; the shared-world adapter clips its output and routes it into
the frozen RoboNaldo locomotion policy.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

REACTIVE_ROUTE_FEATURE_NAMES = (
    "target_dx_m",
    "target_dy_m",
    "self_vx_mps",
    "self_vy_mps",
    "ball_dx_m",
    "ball_dy_m",
    "carrier_dx_m",
    "carrier_dy_m",
    "other_role_dx_m",
    "other_role_dy_m",
    "action_pass",
    "action_shoot",
    "role_teammate",
    "role_defender",
)
_FEATURE_COUNT = len(REACTIVE_ROUTE_FEATURE_NAMES)
_ROLES = {"teammate", "defender"}
_ACTIONS = {"pass", "shoot"}


def reactive_route_features(
    *,
    target_xy_m: NDArray[np.float64],
    self_position_xy_m: NDArray[np.float64],
    self_velocity_xy_mps: NDArray[np.float64],
    ball_position_xy_m: NDArray[np.float64],
    carrier_position_xy_m: NDArray[np.float64],
    other_role_position_xy_m: NDArray[np.float64],
    action: str,
    role: str,
) -> NDArray[np.float64]:
    """Build the exact observation vector used for training and execution."""

    vectors = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (
            target_xy_m,
            self_position_xy_m,
            self_velocity_xy_mps,
            ball_position_xy_m,
            carrier_position_xy_m,
            other_role_position_xy_m,
        )
    )
    if any(value.shape != (2,) or not np.all(np.isfinite(value)) for value in vectors):
        raise ValueError("reactive route observations must contain finite planar vectors")
    if action not in _ACTIONS or role not in _ROLES:
        raise ValueError("reactive route action or role is unsupported")
    target, position, velocity, ball, carrier, other = vectors
    result: NDArray[np.float64] = np.asarray(
        (
            *(target - position),
            *velocity,
            *(ball - position),
            *(carrier - position),
            *(other - position),
            float(action == "pass"),
            float(action == "shoot"),
            float(role == "teammate"),
            float(role == "defender"),
        ),
        dtype=np.float64,
    )
    return result


@dataclass(frozen=True)
class ReactiveRouteSample:
    episode_id: str
    features: tuple[float, ...]
    teacher_world_command_xy_mps: tuple[float, float]

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float64)
        command = np.asarray(self.teacher_world_command_xy_mps, dtype=np.float64)
        if (
            not self.episode_id
            or features.shape != (_FEATURE_COUNT,)
            or command.shape != (2,)
            or not np.all(np.isfinite(features))
            or not np.all(np.isfinite(command))
        ):
            raise ValueError("reactive route sample is malformed")


@dataclass(frozen=True)
class ReactiveRouteDecision:
    accepted: bool
    world_command_xy_mps: tuple[float, float]
    support_distance: float
    actor_hash: str


@dataclass(frozen=True)
class G1ReactiveRouteActor:
    """Observation-conditioned ridge actor with an explicit support gate."""

    source_stage_hash: str
    training_snapshot_hash: str
    implementation_hash: str
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    feature_minimum: tuple[float, ...]
    feature_maximum: tuple[float, ...]
    output_weights: tuple[tuple[float, ...], ...]
    training_episode_ids: tuple[str, ...]
    ridge_l2: float = 1.0
    maximum_ood_distance: float = 2.0
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    schema_version: str = "rosclaw_soccer.g1_reactive_route_actor.v1"

    def __post_init__(self) -> None:
        commitments = (
            self.source_stage_hash,
            self.training_snapshot_hash,
            self.implementation_hash,
        )
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        minimum = np.asarray(self.feature_minimum, dtype=np.float64)
        maximum = np.asarray(self.feature_maximum, dtype=np.float64)
        weights = np.asarray(self.output_weights, dtype=np.float64)
        if any(not value.startswith("sha256:") or len(value) != 71 for value in commitments):
            raise ValueError("reactive route actor commitments must be SHA-256 hashes")
        if (
            center.shape != (_FEATURE_COUNT,)
            or scale.shape != (_FEATURE_COUNT,)
            or minimum.shape != (_FEATURE_COUNT,)
            or maximum.shape != (_FEATURE_COUNT,)
            or weights.shape != (2, _FEATURE_COUNT + 1)
            or not all(
                np.all(np.isfinite(value)) for value in (center, scale, minimum, maximum, weights)
            )
            or np.any(scale <= 0.0)
            or np.any(minimum > maximum)
            or len(self.training_episode_ids) < 8
            or len(set(self.training_episode_ids)) != len(self.training_episode_ids)
        ):
            raise ValueError("reactive route actor support, weights or episodes are invalid")
        if (
            not math.isfinite(self.ridge_l2)
            or not 0.0 < self.ridge_l2 <= 100.0
            or not math.isfinite(self.maximum_ood_distance)
            or not 0.1 <= self.maximum_ood_distance <= 5.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.direct_joint_torque_output
        ):
            raise ValueError("reactive route actor violates its bounded SIM-only contract")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            **asdict(self),
            "feature_names": list(REACTIVE_ROUTE_FEATURE_NAMES),
            "algorithm": "observation_conditioned_ridge_route_v1",
            "pixels_used_for_training": False,
            "current_stage_retention_evidence_used_for_training": False,
            "released_source_stage_evidence_used_for_training": True,
            "output_authority": "BOUNDED_PLANAR_VELOCITY_TO_FROZEN_LOCOMOTION",
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(self, features: NDArray[np.float64]) -> ReactiveRouteDecision:
        observation = np.asarray(features, dtype=np.float64)
        if observation.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(observation)):
            raise ValueError("reactive route actor requires fourteen finite features")
        minimum = np.asarray(self.feature_minimum, dtype=np.float64)
        maximum = np.asarray(self.feature_maximum, dtype=np.float64)
        span = np.maximum(maximum - minimum, 0.10)
        distance = float(
            np.linalg.norm(
                (np.maximum(minimum - observation, 0.0) + np.maximum(observation - maximum, 0.0))
                / span
            )
        )
        if distance > self.maximum_ood_distance:
            return ReactiveRouteDecision(False, (0.0, 0.0), distance, self.actor_hash)
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        design = np.concatenate(((observation - center) / scale, np.ones(1)))
        command = np.asarray(self.output_weights, dtype=np.float64) @ design
        return ReactiveRouteDecision(
            True,
            (float(command[0]), float(command[1])),
            distance,
            self.actor_hash,
        )


def fit_reactive_route_actor(
    samples: Iterable[ReactiveRouteSample],
    *,
    source_stage_hash: str,
    ridge_l2: float = 1.0,
    maximum_ood_distance: float = 2.0,
) -> G1ReactiveRouteActor:
    rows = tuple(samples)
    episode_ids = tuple(sorted({row.episode_id for row in rows}))
    if len(rows) < 1_000 or len(episode_ids) < 8:
        raise ValueError("reactive route actor needs at least 1,000 samples from eight episodes")
    if not source_stage_hash.startswith("sha256:") or len(source_stage_hash) != 71:
        raise ValueError("reactive route source stage must be content bound")
    features = np.asarray([row.features for row in rows], dtype=np.float64)
    targets = np.asarray([row.teacher_world_command_xy_mps for row in rows], dtype=np.float64)
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale < 1.0e-6, 1.0, scale)
    design = np.column_stack(((features - center) / scale, np.ones(len(features))))
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_l2
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets).T
    implementation_hash = hash_bytes(Path(__file__).read_bytes())
    training_snapshot_hash = hash_json(
        {
            "source_stage_hash": source_stage_hash,
            "implementation_hash": implementation_hash,
            "episode_ids": episode_ids,
            "sample_hash": hash_json([asdict(row) for row in rows]),
            "ridge_l2": ridge_l2,
            "maximum_ood_distance": maximum_ood_distance,
        }
    )
    return G1ReactiveRouteActor(
        source_stage_hash=source_stage_hash,
        training_snapshot_hash=training_snapshot_hash,
        implementation_hash=implementation_hash,
        feature_center=tuple(float(value) for value in center),
        feature_scale=tuple(float(value) for value in scale),
        feature_minimum=tuple(float(value) for value in np.min(features, axis=0)),
        feature_maximum=tuple(float(value) for value in np.max(features, axis=0)),
        output_weights=tuple(tuple(float(value) for value in row) for row in weights),
        training_episode_ids=episode_ids,
        ridge_l2=ridge_l2,
        maximum_ood_distance=maximum_ood_distance,
    )


def save_reactive_route_actor(actor: G1ReactiveRouteActor, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_reactive_route_actor(path: Path) -> G1ReactiveRouteActor:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("reactive route actor artifact must be an object")
    claimed = payload.pop("actor_hash", None)
    if payload.pop("feature_names", None) != list(REACTIVE_ROUTE_FEATURE_NAMES):
        raise ValueError("reactive route actor feature contract does not match")
    metadata = {
        "algorithm": "observation_conditioned_ridge_route_v1",
        "pixels_used_for_training": False,
        "current_stage_retention_evidence_used_for_training": False,
        "released_source_stage_evidence_used_for_training": True,
        "output_authority": "BOUNDED_PLANAR_VELOCITY_TO_FROZEN_LOCOMOTION",
    }
    if any(payload.pop(key, None) != value for key, value in metadata.items()):
        raise ValueError("reactive route actor metadata contract does not match")
    for key in (
        "feature_center",
        "feature_scale",
        "feature_minimum",
        "feature_maximum",
        "training_episode_ids",
    ):
        payload[key] = tuple(payload[key])
    payload["output_weights"] = tuple(tuple(row) for row in payload["output_weights"])
    actor = G1ReactiveRouteActor(**payload)
    if claimed != actor.actor_hash:
        raise ValueError("reactive route actor hash does not match its payload")
    return actor


__all__ = [
    "G1ReactiveRouteActor",
    "REACTIVE_ROUTE_FEATURE_NAMES",
    "ReactiveRouteDecision",
    "ReactiveRouteSample",
    "fit_reactive_route_actor",
    "load_reactive_route_actor",
    "reactive_route_features",
    "save_reactive_route_actor",
]
