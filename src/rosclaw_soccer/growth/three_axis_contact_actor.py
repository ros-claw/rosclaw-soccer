"""Evidence-distilled three-axis G1 contact actor.

The operational-space teacher is allowed only while collecting SIM evidence.
This artifact learns the teacher's force response from measured foot velocity
and decodes its own bounded XYZ force through the live MuJoCo Jacobian.  It is
therefore a small, proprioceptive muscle-memory policy rather than a scripted
ball impulse.  The contract is deliberately SIM-only and cannot hot-swap.
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

_SCHEMA = "rosclaw.growth.g1_three_axis_contact_actor.v1"
_FEATURE_NAMES = ("bias", "foot_vx_mps", "foot_vy_mps", "foot_vz_mps")
_OUTPUT_NAMES = ("forward_force_n", "lateral_force_n", "vertical_force_n")


def _commitment(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{label} must be a SHA-256 commitment")
    return value


def _finite_vector(values: object, *, shape: tuple[int, ...], label: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != shape or not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must have finite shape {shape}")
    return vector


@dataclass(frozen=True)
class G1ThreeAxisContactActor:
    """Frozen linear policy with a measured Jacobian-transpose decoder."""

    body_hash: str
    implementation_hash: str
    source_evidence_hashes: tuple[str, ...]
    training_snapshot_hash: str
    task_space_actor_weight_matrix: tuple[tuple[float, ...], ...]
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
        _commitment(self.body_hash, "body_hash")
        _commitment(self.implementation_hash, "implementation_hash")
        _commitment(self.training_snapshot_hash, "training_snapshot_hash")
        if len(self.source_evidence_hashes) < 4 or len(set(self.source_evidence_hashes)) != len(
            self.source_evidence_hashes
        ):
            raise ValueError("three-axis actor requires four unique evidence commitments")
        for index, value in enumerate(self.source_evidence_hashes):
            _commitment(value, f"source_evidence_hashes[{index}]")
        weights = _finite_vector(
            self.task_space_actor_weight_matrix,
            shape=(3, 4),
            label="three-axis actor weights",
        )
        minimum = _finite_vector(
            self.minimum_force_xyz_n,
            shape=(3,),
            label="three-axis actor minimum force",
        )
        maximum = _finite_vector(
            self.maximum_force_xyz_n,
            shape=(3,),
            label="three-axis actor maximum force",
        )
        if (
            self.schema_version != _SCHEMA
            or np.any(minimum < -250.0)
            or np.any(maximum > 250.0)
            or np.any(minimum > maximum)
            or not np.any(np.abs(weights) > 0.0)
            or not 0.15 <= self.maximum_foot_ball_distance_m <= 0.60
            or not 150 <= self.start_policy_frame < self.end_policy_frame <= 430
            or self.training_sample_count < 32
            or self.training_trajectory_count < 4
            or not 1 <= self.failed_trajectory_count < self.training_trajectory_count
            or not math.isfinite(self.distillation_rmse_n)
            or not 0.0 <= self.distillation_rmse_n <= 10.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.online_hot_swap_allowed
        ):
            raise ValueError("three-axis contact actor violates its SIM-only contract")
        offset = _finite_vector(
            self.foot_strike_point_offset_m,
            shape=(3,),
            label="three-axis actor foot strike point",
        )
        if float(np.linalg.norm(offset)) > 0.30:
            raise ValueError("three-axis actor foot strike point is outside the foot envelope")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "source_evidence_hashes": list(self.source_evidence_hashes),
            "task_space_actor_weight_matrix": [
                list(row) for row in self.task_space_actor_weight_matrix
            ],
            "feature_names": list(_FEATURE_NAMES),
            "output_names": list(_OUTPUT_NAMES),
            "algorithm": "ridge_distillation_from_measured_teacher_foot_velocity",
            "decoder": "measured_strike_foot_jacobian_transpose_to_joint_torque",
            "direct_joint_torque_output": True,
            "teacher_required_at_runtime": False,
            "stability_plasticity_contract": {
                "stability": "frozen evidence-bound actor, proximity gate and force clipping",
                "plasticity": "new teacher trajectories create a separate candidate artifact",
            },
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value


@dataclass(frozen=True)
class G1ThreeAxisContactEffect:
    torque: NDArray[np.float64]
    force_xyz_n: NDArray[np.float64]
    foot_velocity_xyz_mps: NDArray[np.float64]
    active: bool
    foot_ball_distance_m: float | None = None


def fit_g1_three_axis_contact_actor(
    *,
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
) -> G1ThreeAxisContactActor:
    """Fit a frozen actor from exact teacher-time proprioception and force."""

    velocity = np.asarray(foot_velocity_xyz_mps, dtype=np.float64)
    force = np.asarray(teacher_force_xyz_n, dtype=np.float64)
    if (
        velocity.ndim != 2
        or velocity.shape[1:] != (3,)
        or force.shape != velocity.shape
        or velocity.shape[0] < 32
        or not np.all(np.isfinite(velocity))
        or not np.all(np.isfinite(force))
        or not math.isfinite(ridge_regularization)
        or not 0.0 < ridge_regularization <= 1.0
    ):
        raise ValueError("three-axis distillation samples or regularization are invalid")
    # The teacher is axis-separable by construction.  Preserve that causal
    # structure instead of allowing correlations in a small rollout batch to
    # create cross-axis force terms and post-contact jitter.
    weights: NDArray[np.float64] = np.zeros((3, 4), dtype=np.float64)
    prediction = np.zeros_like(force)
    for axis in range(3):
        design = np.column_stack((np.ones(len(velocity), dtype=np.float64), velocity[:, axis]))
        penalty = np.diag((0.0, ridge_regularization))
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ force[:, axis])
        weights[axis, 0] = coefficients[0]
        weights[axis, axis + 1] = coefficients[1]
        prediction[:, axis] = design @ coefficients
    prediction = np.clip(prediction, np.min(force, axis=0), np.max(force, axis=0))
    rmse = float(np.sqrt(np.mean(np.square(prediction - force))))
    snapshot = {
        "source_evidence_hashes": list(source_evidence_hashes),
        "velocity_xyz_mps": velocity.tolist(),
        "teacher_force_xyz_n": force.tolist(),
        "ridge_regularization": ridge_regularization,
    }
    return G1ThreeAxisContactActor(
        body_hash=body_hash,
        implementation_hash=implementation_hash,
        source_evidence_hashes=source_evidence_hashes,
        training_snapshot_hash=str(hash_json(snapshot)),
        task_space_actor_weight_matrix=tuple(tuple(float(item) for item in row) for row in weights),
        minimum_force_xyz_n=cast(
            tuple[float, float, float], tuple(float(item) for item in np.min(force, axis=0))
        ),
        maximum_force_xyz_n=cast(
            tuple[float, float, float], tuple(float(item) for item in np.max(force, axis=0))
        ),
        maximum_foot_ball_distance_m=maximum_foot_ball_distance_m,
        start_policy_frame=start_policy_frame,
        end_policy_frame=end_policy_frame,
        foot_strike_point_offset_m=foot_strike_point_offset_m,
        training_sample_count=len(velocity),
        training_trajectory_count=training_trajectory_count,
        failed_trajectory_count=failed_trajectory_count,
        distillation_rmse_n=rmse,
    )


def project_g1_three_axis_contact_actor(
    *,
    jacobian_position: np.ndarray,
    generalized_velocity: np.ndarray,
    actor: G1ThreeAxisContactActor,
    actuated_dof_indices: NDArray[np.int64] | None = None,
    lateral_mirror_sign: float = 1.0,
) -> G1ThreeAxisContactEffect:
    """Evaluate the learned policy and project XYZ force into joint torque."""

    jacobian = np.asarray(jacobian_position, dtype=np.float64)
    velocity = np.asarray(generalized_velocity, dtype=np.float64)
    if (
        jacobian.ndim != 2
        or jacobian.shape[0] != 3
        or jacobian.shape[1] < 35
        or velocity.shape != (jacobian.shape[1],)
    ):
        raise ValueError("three-axis actor expects matching Jacobian/velocity dimensions")
    if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(velocity)):
        raise FloatingPointError("three-axis actor inputs must be finite")
    if lateral_mirror_sign not in {-1.0, 1.0}:
        raise ValueError("three-axis actor mirror sign must be -1 or +1")
    physical_velocity = jacobian @ velocity
    canonical_velocity = physical_velocity.copy()
    canonical_velocity[1] *= lateral_mirror_sign
    features = np.concatenate((np.ones(1, dtype=np.float64), canonical_velocity))
    canonical_force = np.asarray(actor.task_space_actor_weight_matrix, dtype=np.float64) @ features
    canonical_force = np.clip(
        canonical_force,
        np.asarray(actor.minimum_force_xyz_n, dtype=np.float64),
        np.asarray(actor.maximum_force_xyz_n, dtype=np.float64),
    )
    physical_force = canonical_force.copy()
    physical_force[1] *= lateral_mirror_sign
    if actuated_dof_indices is None:
        decoder_indices: NDArray[np.int64] = np.arange(6, 35, dtype=np.int64)
    else:
        decoder_indices = np.asarray(actuated_dof_indices, dtype=np.int64)
        if (
            decoder_indices.shape != (29,)
            or len(np.unique(decoder_indices)) != 29
            or np.any(decoder_indices < 0)
            or np.any(decoder_indices >= jacobian.shape[1])
        ):
            raise ValueError("three-axis actor requires 29 unique actuated DoF indices")
    torque = jacobian[:, decoder_indices].T @ physical_force
    if torque.shape != (29,) or not np.all(np.isfinite(torque)):
        raise FloatingPointError("three-axis actor emitted invalid joint torque")
    return G1ThreeAxisContactEffect(
        torque=torque,
        force_xyz_n=physical_force,
        foot_velocity_xyz_mps=physical_velocity,
        active=bool(np.any(np.abs(physical_force) > 0.0)),
    )


def g1_three_axis_contact_effect(
    *,
    model: Any,
    data: Any,
    right_ankle_body_id: int,
    actor: G1ThreeAxisContactActor,
    policy_frame: int,
    contact_observed: bool,
    ball_position: np.ndarray,
    actuated_dof_indices: NDArray[np.int64] | None = None,
    striking_ankle_body_id: int | None = None,
    lateral_mirror_sign: float = 1.0,
) -> G1ThreeAxisContactEffect:
    """Apply policy/proximity gates before evaluating against live MuJoCo state."""

    zero29: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
    zero3: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
    if contact_observed or not actor.start_policy_frame <= policy_frame <= actor.end_policy_frame:
        return G1ThreeAxisContactEffect(zero29, zero3, zero3, False)
    ball = np.asarray(ball_position, dtype=np.float64)
    if ball.shape != (3,) or not np.all(np.isfinite(ball)):
        raise ValueError("three-axis actor requires a finite ball position")
    ankle_body_id = (
        right_ankle_body_id if striking_ankle_body_id is None else int(striking_ankle_body_id)
    )
    foot_rotation = np.asarray(data.xmat[ankle_body_id], dtype=np.float64).reshape(3, 3)
    foot_point = np.asarray(
        data.xpos[ankle_body_id], dtype=np.float64
    ) + foot_rotation @ np.asarray(actor.foot_strike_point_offset_m, dtype=np.float64)
    distance = float(np.linalg.norm(foot_point - ball))
    if distance > actor.maximum_foot_ball_distance_m:
        return G1ThreeAxisContactEffect(zero29, zero3, zero3, False, distance)

    import mujoco

    jacobian: NDArray[np.float64] = np.zeros((3, int(model.nv)), dtype=np.float64)
    rotation_jacobian: NDArray[np.float64] = np.zeros((3, int(model.nv)), dtype=np.float64)
    mujoco.mj_jac(
        model,
        data,
        jacobian,
        rotation_jacobian,
        foot_point,
        ankle_body_id,
    )
    effect = project_g1_three_axis_contact_actor(
        jacobian_position=jacobian,
        generalized_velocity=np.asarray(data.qvel, dtype=np.float64),
        actor=actor,
        actuated_dof_indices=actuated_dof_indices,
        lateral_mirror_sign=lateral_mirror_sign,
    )
    return G1ThreeAxisContactEffect(
        torque=effect.torque,
        force_xyz_n=effect.force_xyz_n,
        foot_velocity_xyz_mps=effect.foot_velocity_xyz_mps,
        active=effect.active,
        foot_ball_distance_m=distance,
    )


def save_g1_three_axis_contact_actor(actor: G1ThreeAxisContactActor, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_g1_three_axis_contact_actor(path: Path) -> G1ThreeAxisContactActor:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("three-axis contact actor must be an object")
    expected = payload.pop("actor_hash", None)
    for key in (
        "feature_names",
        "output_names",
        "algorithm",
        "decoder",
        "direct_joint_torque_output",
        "teacher_required_at_runtime",
        "stability_plasticity_contract",
    ):
        payload.pop(key, None)
    payload["source_evidence_hashes"] = tuple(payload["source_evidence_hashes"])
    payload["task_space_actor_weight_matrix"] = tuple(
        tuple(float(item) for item in row) for row in payload["task_space_actor_weight_matrix"]
    )
    for key in (
        "minimum_force_xyz_n",
        "maximum_force_xyz_n",
        "foot_strike_point_offset_m",
    ):
        payload[key] = tuple(float(item) for item in payload[key])
    actor = G1ThreeAxisContactActor(**payload)
    if expected != actor.actor_hash:
        raise ValueError("three-axis contact actor hash mismatch")
    return actor


__all__ = [
    "G1ThreeAxisContactActor",
    "G1ThreeAxisContactEffect",
    "g1_three_axis_contact_effect",
    "fit_g1_three_axis_contact_actor",
    "load_g1_three_axis_contact_actor",
    "project_g1_three_axis_contact_actor",
    "save_g1_three_axis_contact_actor",
]
