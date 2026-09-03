"""Target-conditioned task-space muscle memory for G1 contact.

The artifact distils a SIM-only operational-space teacher across different
desired foot velocities.  Runtime inputs are the committed target velocity
and live proprioceptive foot velocity; output force is decoded through the
measured MuJoCo Jacobian into 29 joint torques.  The teacher is never needed
at runtime and the artifact cannot authorize hardware or hot-swap itself.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json

_SCHEMA = "rosclaw.growth.g1_target_velocity_contact_actor.v1"
_FEATURE_NAMES = (
    "bias",
    "target_foot_velocity_axis_mps",
    "measured_foot_velocity_axis_mps",
)


def _commitment(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{label} must be a SHA-256 commitment")
    return value


def _finite(values: object, shape: tuple[int, ...], label: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != shape or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must have finite shape {shape}")
    return vector


@dataclass(frozen=True)
class G1TargetVelocityContactActor:
    body_hash: str
    implementation_hash: str
    source_evidence_hashes: tuple[str, ...]
    training_snapshot_hash: str
    axis_weight_matrix: tuple[tuple[float, float, float], ...]
    minimum_target_velocity_xyz_mps: tuple[float, float, float]
    maximum_target_velocity_xyz_mps: tuple[float, float, float]
    minimum_force_xyz_n: tuple[float, float, float]
    maximum_force_xyz_n: tuple[float, float, float]
    maximum_foot_ball_distance_m: float
    start_policy_frame: int
    end_policy_frame: int
    foot_strike_point_offset_m: tuple[float, float, float]
    training_sample_count: int
    training_trajectory_count: int
    failed_trajectory_count: int
    distillation_rmse_n: float
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    online_hot_swap_allowed: bool = False
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        for value, label in (
            (self.body_hash, "body_hash"),
            (self.implementation_hash, "implementation_hash"),
            (self.training_snapshot_hash, "training_snapshot_hash"),
        ):
            _commitment(value, label)
        if not self.source_evidence_hashes:
            raise ValueError("target-velocity actor needs evidence commitments")
        for index, value in enumerate(self.source_evidence_hashes):
            _commitment(value, f"source_evidence_hashes[{index}]")
        weights = _finite(self.axis_weight_matrix, (3, 3), "axis weights")
        target_min = _finite(self.minimum_target_velocity_xyz_mps, (3,), "minimum target velocity")
        target_max = _finite(self.maximum_target_velocity_xyz_mps, (3,), "maximum target velocity")
        force_min = _finite(self.minimum_force_xyz_n, (3,), "minimum force")
        force_max = _finite(self.maximum_force_xyz_n, (3,), "maximum force")
        offset = _finite(self.foot_strike_point_offset_m, (3,), "foot strike point")
        if (
            self.schema_version != _SCHEMA
            or np.any(target_min > target_max)
            or not 5.0 <= target_min[0] <= target_max[0] <= 20.0
            or target_min[1] > -1.0
            or target_max[1] < 1.0
            or target_min[2] > -0.5
            or target_max[2] < 3.0
            or np.any(force_min < -250.0)
            or np.any(force_max > 250.0)
            or np.any(force_min > force_max)
            or not np.all(np.abs(weights[:, 1]) > 0.0)
            or not np.all(np.abs(weights[:, 2]) > 0.0)
            or not 0.15 <= self.maximum_foot_ball_distance_m <= 0.60
            or not 150 <= self.start_policy_frame < self.end_policy_frame <= 430
            or float(np.linalg.norm(offset)) > 0.30
            or self.training_sample_count < 32
            or self.training_trajectory_count < 6
            or not 1 <= self.failed_trajectory_count < self.training_trajectory_count
            or not math.isfinite(self.distillation_rmse_n)
            or not 0.0 <= self.distillation_rmse_n <= 10.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.online_hot_swap_allowed
        ):
            raise ValueError("target-velocity contact actor violates its SIM-only contract")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "source_evidence_hashes": list(self.source_evidence_hashes),
            "axis_weight_matrix": [list(row) for row in self.axis_weight_matrix],
            "axis_feature_names": list(_FEATURE_NAMES),
            "output_names": ["forward_force_n", "lateral_force_n", "vertical_force_n"],
            "algorithm": "axis_separable_ridge_target_velocity_teacher_distillation",
            "decoder": "measured_strike_foot_jacobian_transpose_to_joint_torque",
            "direct_joint_torque_output": True,
            "teacher_required_at_runtime": False,
            "stability_plasticity_contract": {
                "stability": "content-bound actor, learned target envelope, proximity gate",
                "plasticity": "new success and failure trajectories create a new artifact",
            },
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def target_supported(self, target_velocity_xyz_mps: object) -> bool:
        target = np.asarray(target_velocity_xyz_mps, dtype=np.float64)
        return bool(
            target.shape == (3,)
            and np.all(np.isfinite(target))
            and np.all(target >= np.asarray(self.minimum_target_velocity_xyz_mps))
            and np.all(target <= np.asarray(self.maximum_target_velocity_xyz_mps))
        )


@dataclass(frozen=True)
class G1TargetVelocityContactEffect:
    torque: NDArray[np.float64]
    force_xyz_n: NDArray[np.float64]
    foot_velocity_xyz_mps: NDArray[np.float64]
    target_velocity_xyz_mps: NDArray[np.float64]
    active: bool
    target_supported: bool
    foot_ball_distance_m: float | None = None


def fit_g1_target_velocity_contact_actor(
    *,
    target_velocity_xyz_mps: np.ndarray,
    foot_velocity_xyz_mps: np.ndarray,
    teacher_force_xyz_n: np.ndarray,
    body_hash: str,
    implementation_hash: str,
    source_evidence_hashes: tuple[str, ...],
    training_trajectory_count: int,
    failed_trajectory_count: int,
    maximum_foot_ball_distance_m: float,
    start_policy_frame: int,
    end_policy_frame: int,
    foot_strike_point_offset_m: tuple[float, float, float] = (0.13, 0.0, -0.025),
    ridge_regularization: float = 1.0e-4,
) -> G1TargetVelocityContactActor:
    target = np.asarray(target_velocity_xyz_mps, dtype=np.float64)
    velocity = np.asarray(foot_velocity_xyz_mps, dtype=np.float64)
    force = np.asarray(teacher_force_xyz_n, dtype=np.float64)
    if (
        target.ndim != 2
        or target.shape[1:] != (3,)
        or velocity.shape != target.shape
        or force.shape != target.shape
        or len(target) < 32
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(velocity))
        or not np.all(np.isfinite(force))
        or not math.isfinite(ridge_regularization)
        or not 0.0 < ridge_regularization <= 1.0
    ):
        raise ValueError("target-velocity distillation samples are invalid")
    weights: NDArray[np.float64] = np.zeros((3, 3), dtype=np.float64)
    prediction = np.zeros_like(force)
    for axis in range(3):
        enabled = np.abs(target[:, axis]) > 1.0e-12
        if np.count_nonzero(enabled) < 8 or np.ptp(target[enabled, axis]) <= 1.0e-9:
            raise ValueError("each target-velocity axis needs varied active teacher support")
        design = np.column_stack(
            (
                np.ones(np.count_nonzero(enabled), dtype=np.float64),
                target[enabled, axis],
                velocity[enabled, axis],
            )
        )
        penalty = np.diag((0.0, ridge_regularization, ridge_regularization))
        coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ force[enabled, axis],
        )
        weights[axis] = coefficients
        prediction[enabled, axis] = design @ coefficients
    force_min = np.min(force, axis=0)
    force_max = np.max(force, axis=0)
    prediction = np.clip(prediction, force_min, force_max)
    snapshot = {
        "source_evidence_hashes": list(source_evidence_hashes),
        "target_velocity_xyz_mps": target.tolist(),
        "foot_velocity_xyz_mps": velocity.tolist(),
        "teacher_force_xyz_n": force.tolist(),
        "ridge_regularization": ridge_regularization,
    }
    return G1TargetVelocityContactActor(
        body_hash=body_hash,
        implementation_hash=implementation_hash,
        source_evidence_hashes=source_evidence_hashes,
        training_snapshot_hash=str(hash_json(snapshot)),
        axis_weight_matrix=cast(
            tuple[tuple[float, float, float], ...],
            tuple(tuple(float(item) for item in row) for row in weights),
        ),
        minimum_target_velocity_xyz_mps=cast(
            tuple[float, float, float], tuple(float(item) for item in np.min(target, axis=0))
        ),
        maximum_target_velocity_xyz_mps=cast(
            tuple[float, float, float], tuple(float(item) for item in np.max(target, axis=0))
        ),
        minimum_force_xyz_n=cast(
            tuple[float, float, float], tuple(float(item) for item in force_min)
        ),
        maximum_force_xyz_n=cast(
            tuple[float, float, float], tuple(float(item) for item in force_max)
        ),
        maximum_foot_ball_distance_m=maximum_foot_ball_distance_m,
        start_policy_frame=start_policy_frame,
        end_policy_frame=end_policy_frame,
        foot_strike_point_offset_m=foot_strike_point_offset_m,
        training_sample_count=len(target),
        training_trajectory_count=training_trajectory_count,
        failed_trajectory_count=failed_trajectory_count,
        distillation_rmse_n=float(np.sqrt(np.mean(np.square(prediction - force)))),
    )


def project_g1_target_velocity_contact_actor(
    *,
    jacobian_position: np.ndarray,
    generalized_velocity: np.ndarray,
    target_velocity_xyz_mps: np.ndarray,
    actor: G1TargetVelocityContactActor,
    actuated_dof_indices: NDArray[np.int64] | None = None,
    lateral_mirror_sign: float = 1.0,
) -> G1TargetVelocityContactEffect:
    jacobian = np.asarray(jacobian_position, dtype=np.float64)
    velocity = np.asarray(generalized_velocity, dtype=np.float64)
    target = np.asarray(target_velocity_xyz_mps, dtype=np.float64)
    if (
        jacobian.ndim != 2
        or jacobian.shape[0] != 3
        or jacobian.shape[1] < 35
        or velocity.shape != (jacobian.shape[1],)
        or target.shape != (3,)
        or not np.all(np.isfinite(jacobian))
        or not np.all(np.isfinite(velocity))
        or not np.all(np.isfinite(target))
        or lateral_mirror_sign not in {-1.0, 1.0}
    ):
        raise ValueError("target-velocity actor inputs are invalid")
    canonical_target = target.copy()
    canonical_target[1] *= lateral_mirror_sign
    if not actor.target_supported(canonical_target):
        zero29: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
        zero3: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        return G1TargetVelocityContactEffect(zero29, zero3, zero3, target.copy(), False, False)
    physical_velocity = jacobian @ velocity
    canonical_velocity = physical_velocity.copy()
    canonical_velocity[1] *= lateral_mirror_sign
    weights = np.asarray(actor.axis_weight_matrix, dtype=np.float64)
    canonical_force = np.asarray(
        [
            weights[axis] @ np.asarray((1.0, canonical_target[axis], canonical_velocity[axis]))
            for axis in range(3)
        ],
        dtype=np.float64,
    )
    canonical_force = np.clip(
        canonical_force,
        np.asarray(actor.minimum_force_xyz_n),
        np.asarray(actor.maximum_force_xyz_n),
    )
    # The teacher contract uses an exact zero target to disable an axis rather
    # than to request zero velocity.  Preserve that hybrid semantic explicitly;
    # otherwise a fitted intercept/damping term invents authority that never
    # existed in the source trace.
    canonical_force[np.abs(canonical_target) <= 1.0e-12] = 0.0
    physical_force = canonical_force.copy()
    physical_force[1] *= lateral_mirror_sign
    if actuated_dof_indices is None:
        indices: NDArray[np.int64] = np.arange(6, 35, dtype=np.int64)
    else:
        indices = np.asarray(actuated_dof_indices, dtype=np.int64)
        if (
            indices.shape != (29,)
            or len(np.unique(indices)) != 29
            or np.any(indices < 0)
            or np.any(indices >= jacobian.shape[1])
        ):
            raise ValueError("target-velocity actor needs 29 unique actuated DoF indices")
    torque = jacobian[:, indices].T @ physical_force
    if torque.shape != (29,) or not np.all(np.isfinite(torque)):
        raise FloatingPointError("target-velocity actor emitted invalid torque")
    return G1TargetVelocityContactEffect(
        torque,
        physical_force,
        physical_velocity,
        target.copy(),
        bool(np.any(np.abs(physical_force) > 0.0)),
        True,
    )


def g1_target_velocity_contact_effect(
    *,
    model: Any,
    data: Any,
    right_ankle_body_id: int,
    actor: G1TargetVelocityContactActor,
    target_velocity_xyz_mps: np.ndarray,
    policy_frame: int,
    contact_observed: bool,
    ball_position: np.ndarray,
    actuated_dof_indices: NDArray[np.int64] | None = None,
    striking_ankle_body_id: int | None = None,
    lateral_mirror_sign: float = 1.0,
) -> G1TargetVelocityContactEffect:
    target = np.asarray(target_velocity_xyz_mps, dtype=np.float64)
    zero29: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
    zero3: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
    if contact_observed or not actor.start_policy_frame <= policy_frame <= actor.end_policy_frame:
        return G1TargetVelocityContactEffect(
            zero29, zero3, zero3, target.copy(), False, actor.target_supported(target)
        )
    ball = np.asarray(ball_position, dtype=np.float64)
    if ball.shape != (3,) or not np.all(np.isfinite(ball)):
        raise ValueError("target-velocity actor requires a finite ball position")
    ankle = right_ankle_body_id if striking_ankle_body_id is None else striking_ankle_body_id
    rotation = np.asarray(data.xmat[ankle], dtype=np.float64).reshape(3, 3)
    foot_point = np.asarray(data.xpos[ankle], dtype=np.float64) + rotation @ np.asarray(
        actor.foot_strike_point_offset_m
    )
    distance = float(np.linalg.norm(foot_point - ball))
    if distance > actor.maximum_foot_ball_distance_m:
        return G1TargetVelocityContactEffect(
            zero29,
            zero3,
            zero3,
            target.copy(),
            False,
            actor.target_supported(target),
            distance,
        )
    import mujoco

    jacobian: NDArray[np.float64] = np.zeros((3, int(model.nv)), dtype=np.float64)
    rotation_jacobian: NDArray[np.float64] = np.zeros((3, int(model.nv)), dtype=np.float64)
    mujoco.mj_jac(model, data, jacobian, rotation_jacobian, foot_point, ankle)
    effect = project_g1_target_velocity_contact_actor(
        jacobian_position=jacobian,
        generalized_velocity=np.asarray(data.qvel, dtype=np.float64),
        target_velocity_xyz_mps=target,
        actor=actor,
        actuated_dof_indices=actuated_dof_indices,
        lateral_mirror_sign=lateral_mirror_sign,
    )
    return G1TargetVelocityContactEffect(
        effect.torque,
        effect.force_xyz_n,
        effect.foot_velocity_xyz_mps,
        effect.target_velocity_xyz_mps,
        effect.active,
        effect.target_supported,
        distance,
    )


def save_g1_target_velocity_contact_actor(actor: G1TargetVelocityContactActor, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_g1_target_velocity_contact_actor(path: Path) -> G1TargetVelocityContactActor:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("target-velocity actor must be an object")
    claimed = payload.pop("actor_hash", None)
    for key in (
        "axis_feature_names",
        "output_names",
        "algorithm",
        "decoder",
        "direct_joint_torque_output",
        "teacher_required_at_runtime",
        "stability_plasticity_contract",
    ):
        payload.pop(key, None)
    payload["source_evidence_hashes"] = tuple(payload["source_evidence_hashes"])
    payload["axis_weight_matrix"] = tuple(
        tuple(float(item) for item in row) for row in payload["axis_weight_matrix"]
    )
    for key in (
        "minimum_target_velocity_xyz_mps",
        "maximum_target_velocity_xyz_mps",
        "minimum_force_xyz_n",
        "maximum_force_xyz_n",
        "foot_strike_point_offset_m",
    ):
        payload[key] = tuple(float(item) for item in payload[key])
    actor = G1TargetVelocityContactActor(**payload)
    if claimed != actor.actor_hash:
        raise ValueError("target-velocity actor hash mismatch")
    return actor


__all__ = [
    "G1TargetVelocityContactActor",
    "G1TargetVelocityContactEffect",
    "fit_g1_target_velocity_contact_actor",
    "g1_target_velocity_contact_effect",
    "load_g1_target_velocity_contact_actor",
    "project_g1_target_velocity_contact_actor",
    "save_g1_target_velocity_contact_actor",
]
