"""Bounded proprioceptive neural muscle memory for G1 ball contact.

The actor is distilled from the *total* contact-time residual torque observed
in simulation.  At runtime it replaces that scripted/teacher residual and
directly emits 29 joint torques from a committed task target, contact phase,
ball state, and joint proprioception.  The persisted artifact is JSON and the
runtime evaluator is NumPy-only.
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

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

NEURAL_CONTACT_FEATURE_NAMES = (
    "contact_phase_offset_frames",
    "target_foot_vx_mps",
    "target_foot_vy_mps",
    "target_foot_vz_mps",
    "ball_local_x_m",
    "ball_local_y_m",
    "ball_local_z_m",
    "ball_local_vx_mps",
    "ball_local_vy_mps",
    "ball_local_vz_mps",
    *(f"joint_position_{index}_rad" for index in range(29)),
    *(f"joint_velocity_{index}_rad_s" for index in range(29)),
)
_FEATURE_COUNT = len(NEURAL_CONTACT_FEATURE_NAMES)
_JOINT_COUNT = 29
_SCHEMA = "rosclaw.growth.g1_neural_contact_actor.v1"


def _commitment(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{label} must be a SHA-256 commitment")
    return value


def _finite(value: object, shape: tuple[int, ...], label: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must have finite shape {shape}")
    return array


@dataclass(frozen=True)
class G1NeuralContactActor:
    body_hash: str
    implementation_hash: str
    source_evidence_hashes: tuple[str, ...]
    training_snapshot_hash: str
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    feature_minimum: tuple[float, ...]
    feature_maximum: tuple[float, ...]
    hidden_one_weights: tuple[tuple[float, ...], ...]
    hidden_one_bias: tuple[float, ...]
    hidden_two_weights: tuple[tuple[float, ...], ...]
    hidden_two_bias: tuple[float, ...]
    output_weights: tuple[tuple[float, ...], ...]
    output_bias: tuple[float, ...]
    minimum_torque_nm: tuple[float, ...]
    maximum_torque_nm: tuple[float, ...]
    minimum_target_velocity_xyz_mps: tuple[float, float, float]
    maximum_target_velocity_xyz_mps: tuple[float, float, float]
    minimum_phase_offset_frames: int
    maximum_phase_offset_frames: int
    maximum_normalized_ood_distance: float
    training_sample_count: int
    training_trajectory_count: int
    failed_trajectory_count: int
    training_rmse_nm: float
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
            raise ValueError("neural contact actor needs evidence commitments")
        for index, value in enumerate(self.source_evidence_hashes):
            _commitment(value, f"source_evidence_hashes[{index}]")
        _finite(self.feature_center, (_FEATURE_COUNT,), "feature center")
        scale = _finite(self.feature_scale, (_FEATURE_COUNT,), "feature scale")
        minimum = _finite(self.feature_minimum, (_FEATURE_COUNT,), "feature minimum")
        maximum = _finite(self.feature_maximum, (_FEATURE_COUNT,), "feature maximum")
        hidden_one = np.asarray(self.hidden_one_weights, dtype=np.float64)
        hidden_size = hidden_one.shape[0] if hidden_one.ndim == 2 else 0
        arrays = (
            hidden_one,
            _finite(self.hidden_one_bias, (hidden_size,), "hidden one bias"),
            _finite(
                self.hidden_two_weights,
                (hidden_size, hidden_size),
                "hidden two weights",
            ),
            _finite(self.hidden_two_bias, (hidden_size,), "hidden two bias"),
            _finite(self.output_weights, (_JOINT_COUNT, hidden_size), "output weights"),
            _finite(self.output_bias, (_JOINT_COUNT,), "output bias"),
            _finite(self.minimum_torque_nm, (_JOINT_COUNT,), "minimum torque"),
            _finite(self.maximum_torque_nm, (_JOINT_COUNT,), "maximum torque"),
        )
        target_min = _finite(self.minimum_target_velocity_xyz_mps, (3,), "minimum target velocity")
        target_max = _finite(self.maximum_target_velocity_xyz_mps, (3,), "maximum target velocity")
        torque_min = arrays[-2]
        torque_max = arrays[-1]
        if (
            self.schema_version != _SCHEMA
            or hidden_size < 16
            or hidden_size > 128
            or hidden_one.shape != (hidden_size, _FEATURE_COUNT)
            or not all(np.all(np.isfinite(array)) for array in arrays)
            or np.any(scale <= 0.0)
            or np.any(minimum > maximum)
            or np.any(target_min > target_max)
            or target_min[0] < 5.0
            or target_max[0] > 20.0
            or np.any(torque_min < -120.0)
            or np.any(torque_max > 120.0)
            or np.any(torque_min > torque_max)
            or not -12 <= self.minimum_phase_offset_frames <= -1
            or not 1 <= self.maximum_phase_offset_frames <= 16
            or not 0.05 <= self.maximum_normalized_ood_distance <= 5.0
            or self.training_sample_count < 24
            or self.training_trajectory_count < 2
            or not 1 <= self.failed_trajectory_count < self.training_trajectory_count
            or not math.isfinite(self.training_rmse_nm)
            or not 0.0 <= self.training_rmse_nm <= 10.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.online_hot_swap_allowed
        ):
            raise ValueError("neural contact actor violates its SIM-only contract")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "source_evidence_hashes": list(self.source_evidence_hashes),
            "feature_names": list(NEURAL_CONTACT_FEATURE_NAMES),
            "algorithm": "two_hidden_layer_tanh_total_contact_torque_distillation",
            "runtime": "numpy_only_json_weights",
            "output_authority": "BOUNDED_29_DOF_CONTACT_RESIDUAL_TORQUE",
            "replaces_scripted_contact_torque": True,
            "teacher_required_at_runtime": False,
            "pixels_used_for_training": False,
            "stability_plasticity_contract": {
                "stability": "immutable actor, task envelope, phase gate, OOD fail closed",
                "plasticity": "new evidence creates a separate candidate artifact",
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
class NeuralContactEffect:
    torque: NDArray[np.float64]
    active: bool
    supported: bool
    normalized_ood_distance: float


def neural_contact_features(
    *,
    phase_offset_frames: float,
    target_velocity_xyz_mps: object,
    ball_local_position_m: object,
    ball_local_velocity_mps: object,
    joint_position_rad: object,
    joint_velocity_rad_s: object,
) -> NDArray[np.float64]:
    target = _finite(target_velocity_xyz_mps, (3,), "target velocity")
    ball_position = _finite(ball_local_position_m, (3,), "ball local position")
    ball_velocity = _finite(ball_local_velocity_mps, (3,), "ball local velocity")
    joint_position = _finite(joint_position_rad, (_JOINT_COUNT,), "joint position")
    joint_velocity = _finite(joint_velocity_rad_s, (_JOINT_COUNT,), "joint velocity")
    if not math.isfinite(phase_offset_frames):
        raise ValueError("contact phase offset must be finite")
    return np.concatenate(
        (
            np.asarray((phase_offset_frames,), dtype=np.float64),
            target,
            ball_position,
            ball_velocity,
            joint_position,
            joint_velocity,
        )
    )


def evaluate_neural_contact_actor(
    *, actor: G1NeuralContactActor, features: object
) -> NeuralContactEffect:
    observation = _finite(features, (_FEATURE_COUNT,), "neural contact observation")
    phase = float(observation[0])
    target = observation[1:4]
    minimum = np.asarray(actor.feature_minimum, dtype=np.float64)
    maximum = np.asarray(actor.feature_maximum, dtype=np.float64)
    span = np.maximum(maximum - minimum, 0.05)
    excursion = np.maximum(minimum - observation, 0.0) + np.maximum(observation - maximum, 0.0)
    ood_distance = float(np.linalg.norm(excursion / span) / math.sqrt(_FEATURE_COUNT))
    supported = bool(
        actor.minimum_phase_offset_frames <= phase <= actor.maximum_phase_offset_frames
        and actor.target_supported(target)
        and ood_distance <= actor.maximum_normalized_ood_distance
    )
    if not supported:
        return NeuralContactEffect(
            torque=np.zeros(_JOINT_COUNT, dtype=np.float64),
            active=False,
            supported=False,
            normalized_ood_distance=ood_distance,
        )
    normalized = (observation - np.asarray(actor.feature_center)) / np.asarray(actor.feature_scale)
    hidden_one = np.tanh(
        np.asarray(actor.hidden_one_weights) @ normalized + np.asarray(actor.hidden_one_bias)
    )
    hidden_two = np.tanh(
        np.asarray(actor.hidden_two_weights) @ hidden_one + np.asarray(actor.hidden_two_bias)
    )
    torque = np.asarray(actor.output_weights) @ hidden_two + np.asarray(actor.output_bias)
    torque = np.clip(
        torque,
        np.asarray(actor.minimum_torque_nm),
        np.asarray(actor.maximum_torque_nm),
    )
    if not np.all(np.isfinite(torque)):
        raise FloatingPointError("neural contact actor emitted non-finite torque")
    return NeuralContactEffect(
        torque=torque,
        active=bool(np.max(np.abs(torque)) > 1.0e-6),
        supported=True,
        normalized_ood_distance=ood_distance,
    )


def fit_g1_neural_contact_actor(
    *,
    features: np.ndarray,
    target_torque_nm: np.ndarray,
    target_velocity_xyz_mps: np.ndarray,
    body_hash: str,
    source_evidence_hashes: tuple[str, ...],
    training_trajectory_count: int,
    failed_trajectory_count: int,
    minimum_phase_offset_frames: int = -5,
    maximum_phase_offset_frames: int = 8,
    maximum_normalized_ood_distance: float = 0.75,
    hidden_size: int = 64,
    epochs: int = 2500,
    seed: int = 130,
) -> G1NeuralContactActor:
    observations = np.asarray(features, dtype=np.float64)
    targets = np.asarray(target_torque_nm, dtype=np.float64)
    task_targets = np.asarray(target_velocity_xyz_mps, dtype=np.float64)
    if (
        observations.ndim != 2
        or observations.shape[1] != _FEATURE_COUNT
        or targets.shape != (len(observations), _JOINT_COUNT)
        or task_targets.shape != (len(observations), 3)
        or len(observations) < 24
        or not np.all(np.isfinite(observations))
        or not np.all(np.isfinite(targets))
        or not np.all(np.isfinite(task_targets))
        or not 16 <= hidden_size <= 128
        or not 200 <= epochs <= 10_000
    ):
        raise ValueError("neural contact training tensors are invalid")
    center = np.mean(observations, axis=0)
    scale = np.std(observations, axis=0)
    scale = np.where(scale < 1.0e-5, 1.0, scale)
    normalized = (observations - center) / scale

    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    x = torch.as_tensor(normalized, dtype=torch.float32)
    y = torch.as_tensor(targets, dtype=torch.float32)
    active = np.max(np.abs(targets), axis=1) > 1.0e-5
    sample_weight = torch.as_tensor(np.where(active, 8.0, 1.0)[:, None], dtype=torch.float32)
    model = torch.nn.Sequential(
        torch.nn.Linear(_FEATURE_COUNT, hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, _JOINT_COUNT),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-5)
    for _ in range(epochs):
        prediction = model(x)
        loss = torch.mean(sample_weight * torch.square(prediction - y))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    with torch.no_grad():
        prediction = model(x).numpy().astype(np.float64)
    rmse = float(np.sqrt(np.mean(np.square(prediction - targets))))
    first = cast(Any, model[0])
    second = cast(Any, model[2])
    output = cast(Any, model[4])
    snapshot = {
        "source_evidence_hashes": list(source_evidence_hashes),
        "features_hash": hash_bytes(observations.tobytes()),
        "targets_hash": hash_bytes(targets.tobytes()),
        "task_targets_hash": hash_bytes(task_targets.tobytes()),
        "hidden_size": hidden_size,
        "epochs": epochs,
        "seed": seed,
    }
    return G1NeuralContactActor(
        body_hash=body_hash,
        implementation_hash=str(hash_bytes(Path(__file__).read_bytes())),
        source_evidence_hashes=source_evidence_hashes,
        training_snapshot_hash=str(hash_json(snapshot)),
        feature_center=tuple(float(value) for value in center),
        feature_scale=tuple(float(value) for value in scale),
        feature_minimum=tuple(float(value) for value in np.min(observations, axis=0)),
        feature_maximum=tuple(float(value) for value in np.max(observations, axis=0)),
        hidden_one_weights=tuple(
            tuple(float(value) for value in row)
            for row in first.weight.detach().numpy().astype(np.float64)
        ),
        hidden_one_bias=tuple(
            float(value) for value in first.bias.detach().numpy().astype(np.float64)
        ),
        hidden_two_weights=tuple(
            tuple(float(value) for value in row)
            for row in second.weight.detach().numpy().astype(np.float64)
        ),
        hidden_two_bias=tuple(
            float(value) for value in second.bias.detach().numpy().astype(np.float64)
        ),
        output_weights=tuple(
            tuple(float(value) for value in row)
            for row in output.weight.detach().numpy().astype(np.float64)
        ),
        output_bias=tuple(
            float(value) for value in output.bias.detach().numpy().astype(np.float64)
        ),
        minimum_torque_nm=tuple(float(value) for value in np.min(targets, axis=0)),
        maximum_torque_nm=tuple(float(value) for value in np.max(targets, axis=0)),
        minimum_target_velocity_xyz_mps=cast(
            tuple[float, float, float],
            tuple(float(value) for value in np.min(task_targets, axis=0)),
        ),
        maximum_target_velocity_xyz_mps=cast(
            tuple[float, float, float],
            tuple(float(value) for value in np.max(task_targets, axis=0)),
        ),
        minimum_phase_offset_frames=minimum_phase_offset_frames,
        maximum_phase_offset_frames=maximum_phase_offset_frames,
        maximum_normalized_ood_distance=maximum_normalized_ood_distance,
        training_sample_count=len(observations),
        training_trajectory_count=training_trajectory_count,
        failed_trajectory_count=failed_trajectory_count,
        training_rmse_nm=rmse,
    )


def save_g1_neural_contact_actor(actor: G1NeuralContactActor, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_g1_neural_contact_actor(path: Path) -> G1NeuralContactActor:
    try:
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("neural contact actor artifact is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("neural contact actor artifact must be an object")
    claimed = payload.pop("actor_hash", None)
    metadata = {
        "feature_names": list(NEURAL_CONTACT_FEATURE_NAMES),
        "algorithm": "two_hidden_layer_tanh_total_contact_torque_distillation",
        "runtime": "numpy_only_json_weights",
        "output_authority": "BOUNDED_29_DOF_CONTACT_RESIDUAL_TORQUE",
        "replaces_scripted_contact_torque": True,
        "teacher_required_at_runtime": False,
        "pixels_used_for_training": False,
        "stability_plasticity_contract": {
            "stability": "immutable actor, task envelope, phase gate, OOD fail closed",
            "plasticity": "new evidence creates a separate candidate artifact",
        },
    }
    if any(payload.pop(key, None) != value for key, value in metadata.items()):
        raise ValueError("neural contact actor metadata contract changed")
    vectors = (
        "source_evidence_hashes",
        "feature_center",
        "feature_scale",
        "feature_minimum",
        "feature_maximum",
        "hidden_one_bias",
        "hidden_two_bias",
        "output_bias",
        "minimum_torque_nm",
        "maximum_torque_nm",
        "minimum_target_velocity_xyz_mps",
        "maximum_target_velocity_xyz_mps",
    )
    matrices = (
        "hidden_one_weights",
        "hidden_two_weights",
        "output_weights",
    )
    for key in vectors:
        if not isinstance(payload.get(key), list):
            raise ValueError(f"neural contact actor {key} must be a list")
        payload[key] = tuple(payload[key])
    for key in matrices:
        if not isinstance(payload.get(key), list):
            raise ValueError(f"neural contact actor {key} must be a matrix")
        payload[key] = tuple(tuple(row) for row in payload[key])
    try:
        actor = G1NeuralContactActor(**payload)
    except TypeError as exc:
        raise ValueError("neural contact actor payload is invalid") from exc
    if claimed != actor.actor_hash:
        raise ValueError("neural contact actor hash does not match")
    return actor


__all__ = [
    "G1NeuralContactActor",
    "NeuralContactEffect",
    "evaluate_neural_contact_actor",
    "fit_g1_neural_contact_actor",
    "load_g1_neural_contact_actor",
    "neural_contact_features",
    "save_g1_neural_contact_actor",
]
