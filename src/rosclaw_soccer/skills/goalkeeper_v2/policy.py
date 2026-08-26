"""Portable, content-addressed Goalkeeper V2 actor artifacts.

Training may use torch and privileged critics.  Deployment and strict CPU
MuJoCo evaluation use this small NumPy actor and the causal observation
contract only.  JSON serialization avoids pickle deserialization.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.goalkeeper_v2.observations import GoalkeeperActorObservation


@dataclass(frozen=True)
class GoalkeeperDenseLayer:
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    activation: str
    schema_version: str = "rosclaw_soccer.goalkeeper_dense_layer.v1"

    def __post_init__(self) -> None:
        weights = tuple(tuple(float(value) for value in row) for row in self.weights)
        bias = tuple(float(value) for value in self.bias)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "bias", bias)
        if not weights or not bias:
            raise ValueError("goalkeeper dense layer cannot be empty")
        width = len(weights[0])
        if width != len(bias) or any(len(row) != width for row in weights):
            raise ValueError("goalkeeper dense layer shape is invalid")
        values = (*bias, *(value for row in weights for value in row))
        if any(not math.isfinite(value) for value in values):
            raise ValueError("goalkeeper dense layer contains non-finite weights")
        if self.activation not in {"tanh", "identity"}:
            raise ValueError("goalkeeper dense layer activation is unsupported")

    @property
    def input_size(self) -> int:
        return len(self.weights)

    @property
    def output_size(self) -> int:
        return len(self.bias)


@dataclass(frozen=True)
class GoalkeeperActorArtifact:
    policy_id: str
    generation: int
    parent_policy_hash: str
    body_hash: str
    actor_observation_contract_hash: str
    motion_library_hash: str
    training_run_hash: str
    layers: tuple[GoalkeeperDenseLayer, ...]
    maximum_lateral_speed_mps: float
    maximum_joint_residual_rad: tuple[float, ...]
    operational_space_reach_enabled: bool = False
    operational_space_reach_damping: float = 0.08
    operational_space_reach_gain: float = 0.25
    operational_space_reach_maximum_step_rad: float = 0.05
    operational_space_reach_ramp_sec: float = 0.18
    operational_space_memory_decay: float = 0.75
    operational_space_memory_maximum_rad: float = 0.22
    output_semantics: tuple[str, ...] = (
        "lateral_velocity_fraction",
        *(f"joint_position_residual.{index}" for index in range(29)),
    )
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    deployed_actor_uses_privileged_critic: bool = False
    serialization: str = "JSON_NUMERIC_ONLY"
    schema_version: str = "rosclaw_soccer.goalkeeper_actor_artifact.v1"

    def __post_init__(self) -> None:
        if not self.policy_id.strip() or not 1 <= self.generation <= 1_000_000:
            raise ValueError("goalkeeper actor requires an id and positive generation")
        for value in (
            self.parent_policy_hash,
            self.body_hash,
            self.actor_observation_contract_hash,
            self.motion_library_hash,
            self.training_run_hash,
        ):
            if not value.startswith("sha256:"):
                raise ValueError("goalkeeper actor provenance requires content hashes")
        if len(self.layers) < 2:
            raise ValueError("goalkeeper actor requires at least two dense layers")
        for previous, following in zip(self.layers, self.layers[1:], strict=False):
            if previous.output_size != following.input_size:
                raise ValueError("goalkeeper actor layer dimensions do not compose")
        if self.layers[-1].output_size != 30 or len(self.output_semantics) != 30:
            raise ValueError("goalkeeper actor must output one velocity and 29 joint residuals")
        if self.layers[-1].activation != "tanh":
            raise ValueError("goalkeeper actor output must be bounded by tanh")
        if not math.isfinite(self.maximum_lateral_speed_mps) or not (
            0.10 <= self.maximum_lateral_speed_mps <= 1.0
        ):
            raise ValueError("goalkeeper actor lateral speed limit is outside [0.10, 1.0]")
        if len(self.maximum_joint_residual_rad) != 29 or any(
            not math.isfinite(value) or not 0.0 <= value <= 0.35
            for value in self.maximum_joint_residual_rad
        ):
            raise ValueError("goalkeeper actor joint residual bounds are invalid")
        reach_values = (
            self.operational_space_reach_damping,
            self.operational_space_reach_gain,
            self.operational_space_reach_maximum_step_rad,
            self.operational_space_reach_ramp_sec,
            self.operational_space_memory_decay,
            self.operational_space_memory_maximum_rad,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in reach_values):
            raise ValueError("goalkeeper operational-space reach settings are invalid")
        if not (
            0.02 <= self.operational_space_reach_damping <= 0.30
            and 0.10 <= self.operational_space_reach_gain <= 1.0
            and 0.04 <= self.operational_space_reach_maximum_step_rad <= 0.25
            and 0.04 <= self.operational_space_reach_ramp_sec <= 0.30
            and 0.50 <= self.operational_space_memory_decay <= 0.95
            and 0.08 <= self.operational_space_memory_maximum_rad <= 1.20
        ):
            raise ValueError("goalkeeper operational-space reach settings are outside bounds")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.deployed_actor_uses_privileged_critic
            or self.serialization != "JSON_NUMERIC_ONLY"
        ):
            raise ValueError("goalkeeper actor violates its SIM_ONLY causal boundary")

    @property
    def policy_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            **asdict(self),
            "layers": [asdict(layer) for layer in self.layers],
            "maximum_joint_residual_rad": list(self.maximum_joint_residual_rad),
            "output_semantics": list(self.output_semantics),
        }
        if include_hash:
            value["policy_hash"] = self.policy_hash
        return value


@dataclass(frozen=True)
class GoalkeeperActorAction:
    lateral_velocity_mps: float
    joint_position_residual_rad: tuple[float, ...]
    operational_space_reach_fraction: float
    policy_hash: str
    schema_version: str = "rosclaw_soccer.goalkeeper_actor_action.v1"

    def __post_init__(self) -> None:
        values = (
            self.lateral_velocity_mps,
            self.operational_space_reach_fraction,
            *self.joint_position_residual_rad,
        )
        if len(self.joint_position_residual_rad) != 29 or any(
            not math.isfinite(value) for value in values
        ):
            raise ValueError("goalkeeper actor action must contain finite bounded values")
        if not -1.0 <= self.operational_space_reach_fraction <= 1.0:
            raise ValueError("goalkeeper reach activation must be in [-1, 1]")
        if not self.policy_hash.startswith("sha256:"):
            raise ValueError("goalkeeper actor action requires a policy hash")


class NumpyGoalkeeperActor:
    """Inference-only actor; it has no transport or hardware authority."""

    def __init__(self, artifact: GoalkeeperActorArtifact) -> None:
        self.artifact = artifact
        self._layers = tuple(
            (
                np.asarray(layer.weights, dtype=np.float64),
                np.asarray(layer.bias, dtype=np.float64),
                layer.activation,
            )
            for layer in artifact.layers
        )

    def action(self, observation: GoalkeeperActorObservation) -> GoalkeeperActorAction:
        if observation.actor_contract_hash != self.artifact.actor_observation_contract_hash:
            raise ValueError("goalkeeper actor observation contract hash mismatch")
        value = np.asarray(observation.values, dtype=np.float64)
        if value.shape != (self.artifact.layers[0].input_size,) or not np.all(np.isfinite(value)):
            raise ValueError("goalkeeper actor input shape or values are invalid")
        for weights, bias, activation in self._layers:
            value = value @ weights + bias
            if activation == "tanh":
                value = np.tanh(value)
        if value.shape != (30,) or not np.all(np.isfinite(value)):
            raise RuntimeError("goalkeeper actor produced an invalid action")
        residual_limit = np.asarray(self.artifact.maximum_joint_residual_rad, dtype=np.float64)
        residual = value[1:] * residual_limit
        # Reaching is a learned intent, not a synonym for lateral velocity.
        # Center blocks and nearly stationary high reaches can have little
        # lateral command while the arm channels are strongly active.
        reach_fraction = max(
            abs(float(value[0])),
            float(np.max(np.abs(value[1 + 15 : 1 + 29]))),
        )
        return GoalkeeperActorAction(
            lateral_velocity_mps=float(value[0] * self.artifact.maximum_lateral_speed_mps),
            joint_position_residual_rad=tuple(float(item) for item in residual),
            operational_space_reach_fraction=reach_fraction,
            policy_hash=self.artifact.policy_hash,
        )


def save_goalkeeper_actor_artifact(
    artifact: GoalkeeperActorArtifact,
    path: Path,
    *,
    source_checkout: Path,
) -> None:
    output = path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("goalkeeper actor artifact must remain outside the source checkout")
    if output.exists():
        raise ValueError("goalkeeper actor artifact output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_goalkeeper_actor_artifact(path: Path) -> GoalkeeperActorArtifact:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    claimed_hash = str(payload.pop("policy_hash", ""))
    try:
        payload["layers"] = tuple(GoalkeeperDenseLayer(**layer) for layer in payload["layers"])
        payload["maximum_joint_residual_rad"] = tuple(payload["maximum_joint_residual_rad"])
        payload["output_semantics"] = tuple(payload["output_semantics"])
        artifact = GoalkeeperActorArtifact(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("goalkeeper actor artifact payload is invalid") from exc
    if claimed_hash != artifact.policy_hash:
        raise ValueError("goalkeeper actor artifact content hash mismatch")
    return artifact


__all__ = [
    "GoalkeeperActorAction",
    "GoalkeeperActorArtifact",
    "GoalkeeperDenseLayer",
    "NumpyGoalkeeperActor",
    "load_goalkeeper_actor_artifact",
    "save_goalkeeper_actor_artifact",
]
