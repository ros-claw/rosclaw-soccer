"""Content-bound temporal route actor for SIM-only team movement.

The model is deliberately small enough to audit and replay with NumPy.  A
training host may use PyTorch, but the persisted artifact contains only JSON
numbers and the runtime never deserializes executable code.  Its output is a
bounded planar velocity request to the frozen locomotion policy; it has no
joint, torque, pose, ball, ROS, DDS or hardware authority.
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

from rosclaw_soccer.growth.reactive_route_actor import (
    REACTIVE_ROUTE_FEATURE_NAMES,
    G1ReactiveRouteActor,
    load_reactive_route_actor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_CURRENT_FEATURE_COUNT = len(REACTIVE_ROUTE_FEATURE_NAMES)
TEMPORAL_ROUTE_INPUT_NAMES = (
    *(f"current.{name}" for name in REACTIVE_ROUTE_FEATURE_NAMES),
    *(f"ema.{name}" for name in REACTIVE_ROUTE_FEATURE_NAMES),
    "previous_command_x_mps",
    "previous_command_y_mps",
)
_INPUT_COUNT = len(TEMPORAL_ROUTE_INPUT_NAMES)


@dataclass(frozen=True)
class TemporalRouteSequence:
    """One ordered physical trace; sequence boundaries reset actor memory."""

    sequence_id: str
    features: tuple[tuple[float, ...], ...]
    teacher_world_commands_xy_mps: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        observations = np.asarray(self.features, dtype=np.float64)
        commands = np.asarray(self.teacher_world_commands_xy_mps, dtype=np.float64)
        if (
            not self.sequence_id
            or observations.ndim != 2
            or observations.shape[1:] != (_CURRENT_FEATURE_COUNT,)
            or commands.shape != (observations.shape[0], 2)
            or observations.shape[0] < 32
            or not np.all(np.isfinite(observations))
            or not np.all(np.isfinite(commands))
        ):
            raise ValueError("temporal route sequence is malformed")


@dataclass(frozen=True)
class TemporalRouteMemory:
    feature_ema: tuple[float, ...]
    previous_command_xy_mps: tuple[float, float]

    def __post_init__(self) -> None:
        feature_ema = np.asarray(self.feature_ema, dtype=np.float64)
        command = np.asarray(self.previous_command_xy_mps, dtype=np.float64)
        if (
            feature_ema.shape != (_CURRENT_FEATURE_COUNT,)
            or command.shape != (2,)
            or not np.all(np.isfinite(feature_ema))
            or not np.all(np.isfinite(command))
        ):
            raise ValueError("temporal route memory is malformed")


@dataclass(frozen=True)
class TemporalRouteDecision:
    accepted: bool
    world_command_xy_mps: tuple[float, float]
    support_distance: float
    actor_hash: str
    next_memory: TemporalRouteMemory


@dataclass(frozen=True)
class G1TemporalRouteActor:
    """Two-layer temporal MLP with an explicit raw-observation support gate."""

    source_stage_hash: str
    source_actor_hash: str
    training_snapshot_hash: str
    implementation_hash: str
    input_center: tuple[float, ...]
    input_scale: tuple[float, ...]
    current_feature_minimum: tuple[float, ...]
    current_feature_maximum: tuple[float, ...]
    parent_feature_center: tuple[float, ...]
    parent_feature_scale: tuple[float, ...]
    parent_output_weights: tuple[tuple[float, ...], ...]
    hidden_weights: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[tuple[float, ...], ...]
    output_bias: tuple[float, float]
    training_sequence_ids: tuple[str, ...]
    training_rmse_mps: float
    temporal_ema_alpha: float = 0.22
    output_ceiling_mps: float = 0.65
    residual_ceiling_mps: float = 0.12
    maximum_parent_attenuation_fraction: float = 0.10
    maximum_residual_to_parent_fraction: float = 0.18
    minimum_pass_teammate_temporal_error_m: float = 0.55
    maximum_ood_distance: float = 2.0
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    schema_version: str = "rosclaw_soccer.g1_temporal_route_actor.v1"

    def __post_init__(self) -> None:
        commitments = (
            self.source_stage_hash,
            self.source_actor_hash,
            self.training_snapshot_hash,
            self.implementation_hash,
        )
        center = np.asarray(self.input_center, dtype=np.float64)
        scale = np.asarray(self.input_scale, dtype=np.float64)
        minimum = np.asarray(self.current_feature_minimum, dtype=np.float64)
        maximum = np.asarray(self.current_feature_maximum, dtype=np.float64)
        parent_center = np.asarray(self.parent_feature_center, dtype=np.float64)
        parent_scale = np.asarray(self.parent_feature_scale, dtype=np.float64)
        parent_weights = np.asarray(self.parent_output_weights, dtype=np.float64)
        hidden_weights = np.asarray(self.hidden_weights, dtype=np.float64)
        hidden_bias = np.asarray(self.hidden_bias, dtype=np.float64)
        output_weights = np.asarray(self.output_weights, dtype=np.float64)
        output_bias = np.asarray(self.output_bias, dtype=np.float64)
        hidden_size = hidden_weights.shape[0] if hidden_weights.ndim == 2 else 0
        if any(not value.startswith("sha256:") or len(value) != 71 for value in commitments):
            raise ValueError("temporal route actor commitments must be SHA-256 hashes")
        arrays = (
            center,
            scale,
            minimum,
            maximum,
            parent_center,
            parent_scale,
            parent_weights,
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
        )
        if (
            center.shape != (_INPUT_COUNT,)
            or scale.shape != (_INPUT_COUNT,)
            or minimum.shape != (_CURRENT_FEATURE_COUNT,)
            or maximum.shape != (_CURRENT_FEATURE_COUNT,)
            or parent_center.shape != (_CURRENT_FEATURE_COUNT,)
            or parent_scale.shape != (_CURRENT_FEATURE_COUNT,)
            or parent_weights.shape != (2, _CURRENT_FEATURE_COUNT + 1)
            or hidden_size < 8
            or hidden_weights.shape != (hidden_size, _INPUT_COUNT)
            or hidden_bias.shape != (hidden_size,)
            or output_weights.shape != (2, hidden_size)
            or output_bias.shape != (2,)
            or not all(np.all(np.isfinite(array)) for array in arrays)
            or np.any(scale <= 0.0)
            or np.any(parent_scale <= 0.0)
            or np.any(minimum > maximum)
            or len(self.training_sequence_ids) < 8
            or len(set(self.training_sequence_ids)) != len(self.training_sequence_ids)
        ):
            raise ValueError("temporal route actor tensors or support are invalid")
        values = (
            self.training_rmse_mps,
            self.temporal_ema_alpha,
            self.output_ceiling_mps,
            self.residual_ceiling_mps,
            self.maximum_parent_attenuation_fraction,
            self.maximum_residual_to_parent_fraction,
            self.minimum_pass_teammate_temporal_error_m,
            self.maximum_ood_distance,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or not 0.0 <= self.training_rmse_mps <= 0.25
            or not 0.05 <= self.temporal_ema_alpha <= 0.80
            or not 0.20 <= self.output_ceiling_mps <= 0.70
            or not 0.01 <= self.residual_ceiling_mps <= 0.20
            or not 0.0 <= self.maximum_parent_attenuation_fraction <= 0.25
            or not 0.05 <= self.maximum_residual_to_parent_fraction <= 0.40
            or not 0.30 <= self.minimum_pass_teammate_temporal_error_m <= 0.80
            or not 0.1 <= self.maximum_ood_distance <= 5.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.direct_joint_torque_output
        ):
            raise ValueError("temporal route actor violates its bounded SIM-only contract")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            **asdict(self),
            "input_names": list(TEMPORAL_ROUTE_INPUT_NAMES),
            "algorithm": "temporal_context_mlp_behavior_cloning_v1",
            "serialized_executable_code": False,
            "pixels_used_for_training": False,
            "current_stage_retention_evidence_used_for_training": False,
            "released_source_stage_evidence_used_for_training": True,
            "output_authority": "BOUNDED_PLANAR_VELOCITY_TO_FROZEN_LOCOMOTION",
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(
        self,
        features: NDArray[np.float64],
        memory: TemporalRouteMemory | None = None,
    ) -> TemporalRouteDecision:
        observation = np.asarray(features, dtype=np.float64)
        if observation.shape != (_CURRENT_FEATURE_COUNT,) or not np.all(np.isfinite(observation)):
            raise ValueError("temporal route actor requires fourteen finite features")
        if memory is None:
            previous_ema = observation
            previous_command = np.zeros(2, dtype=np.float64)
        else:
            previous_ema = np.asarray(memory.feature_ema, dtype=np.float64)
            previous_command = np.asarray(memory.previous_command_xy_mps, dtype=np.float64)
        ema = self.temporal_ema_alpha * observation + (1.0 - self.temporal_ema_alpha) * previous_ema
        minimum = np.asarray(self.current_feature_minimum, dtype=np.float64)
        maximum = np.asarray(self.current_feature_maximum, dtype=np.float64)
        span = np.maximum(maximum - minimum, 0.10)
        support_distance = float(
            np.linalg.norm(
                (np.maximum(minimum - observation, 0.0) + np.maximum(observation - maximum, 0.0))
                / span
            )
        )
        temporal_input = np.concatenate((observation, ema, previous_command))
        normalized = (temporal_input - np.asarray(self.input_center)) / np.asarray(self.input_scale)
        hidden = np.tanh(
            np.asarray(self.hidden_weights, dtype=np.float64) @ normalized
            + np.asarray(self.hidden_bias, dtype=np.float64)
        )
        residual = self.residual_ceiling_mps * np.tanh(
            np.asarray(self.output_weights, dtype=np.float64) @ hidden
            + np.asarray(self.output_bias, dtype=np.float64)
        )
        parent_design = np.concatenate(
            (
                (observation - np.asarray(self.parent_feature_center, dtype=np.float64))
                / np.asarray(self.parent_feature_scale, dtype=np.float64),
                np.ones(1, dtype=np.float64),
            )
        )
        parent_command = np.asarray(self.parent_output_weights, dtype=np.float64) @ parent_design
        for axis in range(2):
            if abs(parent_command[axis]) >= 0.08 and residual[axis] * parent_command[axis] < 0.0:
                maximum_attenuation = self.maximum_parent_attenuation_fraction * abs(
                    parent_command[axis]
                )
                residual[axis] = float(
                    np.clip(residual[axis], -maximum_attenuation, maximum_attenuation)
                )
        residual_norm = float(np.linalg.norm(residual))
        residual_budget = max(
            0.015,
            self.maximum_residual_to_parent_fraction * float(np.linalg.norm(parent_command)),
        )
        if residual_norm > residual_budget:
            residual *= residual_budget / residual_norm
        pass_teammate = bool(observation[10] > 0.5 and observation[12] > 0.5)
        if pass_teammate and float(np.linalg.norm(observation[:2])) < (
            self.minimum_pass_teammate_temporal_error_m
        ):
            residual.fill(0.0)
        command = parent_command + residual
        speed = float(np.linalg.norm(command))
        if speed > self.output_ceiling_mps:
            command *= self.output_ceiling_mps / speed
        accepted = support_distance <= self.maximum_ood_distance
        if not accepted:
            command.fill(0.0)
        next_memory = TemporalRouteMemory(
            feature_ema=tuple(float(value) for value in ema),
            previous_command_xy_mps=(float(command[0]), float(command[1])),
        )
        return TemporalRouteDecision(
            accepted=accepted,
            world_command_xy_mps=(float(command[0]), float(command[1])),
            support_distance=support_distance,
            actor_hash=self.actor_hash,
            next_memory=next_memory,
        )


def fit_temporal_route_actor(
    sequences: tuple[TemporalRouteSequence, ...],
    *,
    source_stage_hash: str,
    source_actor_hash: str,
    parent_actor: G1ReactiveRouteActor,
    seed: int = 123,
    hidden_size: int = 32,
    epochs: int = 800,
    learning_rate: float = 3.0e-3,
    weight_decay: float = 1.0e-5,
    temporal_ema_alpha: float = 0.22,
    output_ceiling_mps: float = 0.65,
    residual_ceiling_mps: float = 0.12,
    maximum_parent_attenuation_fraction: float = 0.10,
    maximum_residual_to_parent_fraction: float = 0.18,
    minimum_pass_teammate_temporal_error_m: float = 0.55,
    maximum_ood_distance: float = 2.0,
    device: str = "cpu",
) -> G1TemporalRouteActor:
    """Fit a deterministic candidate; PyTorch is used only on the training host."""

    if len(sequences) < 8 or len({item.sequence_id for item in sequences}) != len(sequences):
        raise ValueError("temporal route training needs at least eight unique sequences")
    if any(
        not value.startswith("sha256:") or len(value) != 71
        for value in (source_stage_hash, source_actor_hash)
    ):
        raise ValueError("temporal route source must be content bound")
    if parent_actor.actor_hash != source_actor_hash:
        raise ValueError("temporal route parent actor does not match its source commitment")
    if not 8 <= hidden_size <= 128 or not 50 <= epochs <= 10_000:
        raise ValueError("temporal route training shape or duration is invalid")
    if not 0.05 <= temporal_ema_alpha <= 0.80:
        raise ValueError("temporal route EMA alpha is invalid")
    inputs, targets = _training_arrays(sequences, temporal_ema_alpha=temporal_ema_alpha)
    center = np.mean(inputs, axis=0)
    scale = np.std(inputs, axis=0)
    scale = np.where(scale < 1.0e-6, 1.0, scale)
    normalized = (inputs - center) / scale

    import torch

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    training_device = torch.device(device)
    x = torch.as_tensor(normalized, dtype=torch.float32, device=training_device)
    current_features = inputs[:, :_CURRENT_FEATURE_COUNT]
    parent_design = np.column_stack(
        (
            (current_features - np.asarray(parent_actor.feature_center, dtype=np.float64))
            / np.asarray(parent_actor.feature_scale, dtype=np.float64),
            np.ones(len(current_features), dtype=np.float64),
        )
    )
    parent_commands = parent_design @ np.asarray(parent_actor.output_weights, dtype=np.float64).T
    residual_targets = np.clip(
        (targets - parent_commands) / residual_ceiling_mps,
        -0.999,
        0.999,
    )
    y = torch.as_tensor(residual_targets, dtype=torch.float32, device=training_device)
    model = torch.nn.Sequential(
        torch.nn.Linear(_INPUT_COUNT, hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, 2),
        torch.nn.Tanh(),
    ).to(training_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    for _ in range(epochs):
        prediction = model(x)
        loss = torch.mean((prediction - y) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    with torch.no_grad():
        residual = model(x).detach().cpu().numpy() * residual_ceiling_mps
        predicted = parent_commands + residual
    rmse = float(np.sqrt(np.mean(np.square(predicted - targets))))
    first = cast(Any, model[0])
    second = cast(Any, model[2])
    hidden_weights = first.weight.detach().cpu().numpy().astype(np.float64)
    hidden_bias = first.bias.detach().cpu().numpy().astype(np.float64)
    output_weights = second.weight.detach().cpu().numpy().astype(np.float64)
    output_bias = second.bias.detach().cpu().numpy().astype(np.float64)
    current = np.concatenate(
        [np.asarray(sequence.features, dtype=np.float64) for sequence in sequences], axis=0
    )
    implementation_hash = hash_bytes(Path(__file__).read_bytes())
    snapshot_hash = hash_json(
        {
            "source_stage_hash": source_stage_hash,
            "source_actor_hash": source_actor_hash,
            "implementation_hash": implementation_hash,
            "sequence_ids": [sequence.sequence_id for sequence in sequences],
            "sequence_hash": hash_json([asdict(sequence) for sequence in sequences]),
            "seed": seed,
            "hidden_size": hidden_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "temporal_ema_alpha": temporal_ema_alpha,
            "output_ceiling_mps": output_ceiling_mps,
            "residual_ceiling_mps": residual_ceiling_mps,
            "maximum_parent_attenuation_fraction": maximum_parent_attenuation_fraction,
            "maximum_residual_to_parent_fraction": maximum_residual_to_parent_fraction,
            "minimum_pass_teammate_temporal_error_m": minimum_pass_teammate_temporal_error_m,
        }
    )
    return G1TemporalRouteActor(
        source_stage_hash=source_stage_hash,
        source_actor_hash=source_actor_hash,
        training_snapshot_hash=snapshot_hash,
        implementation_hash=implementation_hash,
        input_center=tuple(float(value) for value in center),
        input_scale=tuple(float(value) for value in scale),
        current_feature_minimum=tuple(float(value) for value in np.min(current, axis=0)),
        current_feature_maximum=tuple(float(value) for value in np.max(current, axis=0)),
        parent_feature_center=parent_actor.feature_center,
        parent_feature_scale=parent_actor.feature_scale,
        parent_output_weights=parent_actor.output_weights,
        hidden_weights=tuple(tuple(float(value) for value in row) for row in hidden_weights),
        hidden_bias=tuple(float(value) for value in hidden_bias),
        output_weights=tuple(tuple(float(value) for value in row) for row in output_weights),
        output_bias=(float(output_bias[0]), float(output_bias[1])),
        training_sequence_ids=tuple(sorted(sequence.sequence_id for sequence in sequences)),
        training_rmse_mps=rmse,
        temporal_ema_alpha=temporal_ema_alpha,
        output_ceiling_mps=output_ceiling_mps,
        residual_ceiling_mps=residual_ceiling_mps,
        maximum_parent_attenuation_fraction=maximum_parent_attenuation_fraction,
        maximum_residual_to_parent_fraction=maximum_residual_to_parent_fraction,
        minimum_pass_teammate_temporal_error_m=minimum_pass_teammate_temporal_error_m,
        maximum_ood_distance=maximum_ood_distance,
    )


def save_temporal_route_actor(actor: G1TemporalRouteActor, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_temporal_route_actor(path: Path) -> G1TemporalRouteActor:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("temporal route actor artifact must be an object")
    claimed = payload.pop("actor_hash", None)
    if payload.pop("input_names", None) != list(TEMPORAL_ROUTE_INPUT_NAMES):
        raise ValueError("temporal route actor input contract does not match")
    metadata = {
        "algorithm": "temporal_context_mlp_behavior_cloning_v1",
        "serialized_executable_code": False,
        "pixels_used_for_training": False,
        "current_stage_retention_evidence_used_for_training": False,
        "released_source_stage_evidence_used_for_training": True,
        "output_authority": "BOUNDED_PLANAR_VELOCITY_TO_FROZEN_LOCOMOTION",
    }
    if any(payload.pop(key, None) != value for key, value in metadata.items()):
        raise ValueError("temporal route actor metadata contract does not match")
    for key in (
        "input_center",
        "input_scale",
        "current_feature_minimum",
        "current_feature_maximum",
        "parent_feature_center",
        "parent_feature_scale",
        "hidden_bias",
        "output_bias",
        "training_sequence_ids",
    ):
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(f"temporal route actor {key} must be a list")
        payload[key] = tuple(value)
    for key in ("parent_output_weights", "hidden_weights", "output_weights"):
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(f"temporal route actor {key} must be a matrix")
        payload[key] = tuple(tuple(row) for row in value)
    try:
        actor = G1TemporalRouteActor(**payload)
    except TypeError as exc:
        raise ValueError("temporal route actor payload is invalid") from exc
    if claimed != actor.actor_hash:
        raise ValueError("temporal route actor hash does not match")
    return actor


RouteActor = G1ReactiveRouteActor | G1TemporalRouteActor


def load_route_actor(path: Path) -> RouteActor:
    """Load either released route generation without weakening its own validator."""

    source = path.expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("route actor artifact is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("route actor artifact must be an object")
    schema = payload.get("schema_version")
    if schema == "rosclaw_soccer.g1_temporal_route_actor.v1":
        return load_temporal_route_actor(source)
    if schema == "rosclaw_soccer.g1_reactive_route_actor.v1":
        return load_reactive_route_actor(source)
    raise ValueError("route actor schema is unsupported")


def _training_arrays(
    sequences: tuple[TemporalRouteSequence, ...],
    *,
    temporal_ema_alpha: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rows: list[NDArray[np.float64]] = []
    targets: list[NDArray[np.float64]] = []
    for sequence in sequences:
        features = np.asarray(sequence.features, dtype=np.float64)
        commands = np.asarray(sequence.teacher_world_commands_xy_mps, dtype=np.float64)
        ema = features[0].copy()
        previous_command = np.zeros(2, dtype=np.float64)
        for observation, command in zip(features, commands, strict=True):
            ema = temporal_ema_alpha * observation + (1.0 - temporal_ema_alpha) * ema
            rows.append(np.concatenate((observation, ema, previous_command)))
            targets.append(command)
            previous_command = command
    return np.asarray(rows, dtype=np.float64), np.asarray(targets, dtype=np.float64)


__all__ = [
    "G1TemporalRouteActor",
    "TEMPORAL_ROUTE_INPUT_NAMES",
    "TemporalRouteDecision",
    "TemporalRouteMemory",
    "TemporalRouteSequence",
    "fit_temporal_route_actor",
    "load_route_actor",
    "load_temporal_route_actor",
    "save_temporal_route_actor",
]
