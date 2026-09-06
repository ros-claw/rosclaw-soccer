"""Content-bound causal hand-off actor for composed SIM-only skills.

The actor predicts *when* a successor skill may enter.  It never predicts a
joint target, torque, pose or ball state.  A training host may use PyTorch,
while the persisted runtime artifact is JSON and executes with NumPy only.

This narrow contract is useful beyond soccer: a frozen parent skill remains
the fallback, the learned residual can move its trigger only by a bounded
number of control frames, and an out-of-support observation removes all
learned authority.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

CAUSAL_TRANSITION_FEATURE_NAMES = (
    "receiver_to_predecessor_x_m",
    "receiver_to_predecessor_y_m",
    "receiver_ball_x_m",
    "receiver_ball_y_m",
    "receiver_ball_z_m",
    "receiver_reception_target_x_m",
    "receiver_reception_target_y_m",
    "receiver_shot_target_y_m",
    "receiver_shot_target_z_m",
    "predecessor_swing_speed_scale",
    "ball_ground_friction",
    "predecessor_yaw_sin",
    "predecessor_yaw_cos",
    "receiver_kick_foot_sign",
)
_FEATURE_COUNT = len(CAUSAL_TRANSITION_FEATURE_NAMES)


@dataclass(frozen=True)
class CausalTransitionSample:
    """One best-safe hand-off label selected from a MuJoCo timing sweep."""

    sample_id: str
    features: tuple[float, ...]
    optimal_trigger_policy_frame: int
    source_trajectory_hash: str
    safe: bool
    schema_version: str = "rosclaw.growth.causal_transition_sample.v2"

    def __post_init__(self) -> None:
        values = np.asarray(self.features, dtype=np.float64)
        if (
            not self.sample_id
            or not isinstance(self.sample_id, str)
            or values.shape != (_FEATURE_COUNT,)
            or not np.all(np.isfinite(values))
            or isinstance(self.optimal_trigger_policy_frame, bool)
            or not 70 <= self.optimal_trigger_policy_frame <= 125
            or not isinstance(self.source_trajectory_hash, str)
            or not self.source_trajectory_hash.startswith("sha256:")
            or len(self.source_trajectory_hash) != 71
            or type(self.safe) is not bool
            or self.schema_version != "rosclaw.growth.causal_transition_sample.v2"
        ):
            raise ValueError("causal transition sample is malformed")


@dataclass(frozen=True)
class CausalTransitionDecision:
    accepted: bool
    trigger_policy_frame: int
    parent_trigger_policy_frame: int
    residual_frames: int
    support_distance: float
    actor_hash: str
    predicted_safe_probability: float | None = None
    predicted_chain_probability: float | None = None
    ensemble_probability_spread: float | None = None
    used_parent_fallback: bool = False


class CausalSkillTransitionActor(Protocol):
    """Structural runtime contract shared by regression, risk and memory actors."""

    @property
    def implementation_hash(self) -> str: ...

    @property
    def actor_hash(self) -> str: ...

    def decide(self, features: NDArray[np.float64]) -> CausalTransitionDecision: ...


@dataclass(frozen=True)
class G1CausalSkillTransitionActor:
    """Bounded MLP residual around a frozen parent hand-off phase."""

    source_stage_hash: str
    training_snapshot_hash: str
    implementation_hash: str
    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    feature_minimum: tuple[float, ...]
    feature_maximum: tuple[float, ...]
    hidden_weights: tuple[tuple[float, ...], ...]
    hidden_bias: tuple[float, ...]
    output_weights: tuple[float, ...]
    output_bias: float
    training_sample_ids: tuple[str, ...]
    training_rmse_frames: float
    training_trigger_minimum: int
    training_trigger_maximum: int
    parent_trigger_policy_frame: int = 88
    maximum_trigger_residual_frames: int = 15
    minimum_trigger_policy_frame: int = 75
    maximum_trigger_policy_frame: int = 120
    maximum_ood_distance: float = 2.0
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    direct_joint_torque_output: bool = False
    schema_version: str = "rosclaw.growth.g1_causal_skill_transition_actor.v3"

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
        hidden_weights = np.asarray(self.hidden_weights, dtype=np.float64)
        hidden_bias = np.asarray(self.hidden_bias, dtype=np.float64)
        output_weights = np.asarray(self.output_weights, dtype=np.float64)
        hidden_size = hidden_weights.shape[0] if hidden_weights.ndim == 2 else 0
        if any(
            not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71
            for value in commitments
        ):
            raise ValueError("causal transition commitments must be SHA-256 hashes")
        arrays = (center, scale, minimum, maximum, hidden_weights, hidden_bias, output_weights)
        if (
            center.shape != (_FEATURE_COUNT,)
            or scale.shape != (_FEATURE_COUNT,)
            or minimum.shape != (_FEATURE_COUNT,)
            or maximum.shape != (_FEATURE_COUNT,)
            or hidden_size < 4
            or hidden_weights.shape != (hidden_size, _FEATURE_COUNT)
            or hidden_bias.shape != (hidden_size,)
            or output_weights.shape != (hidden_size,)
            or not all(np.all(np.isfinite(value)) for value in arrays)
            or not math.isfinite(self.output_bias)
            or np.any(scale <= 0.0)
            or np.any(minimum > maximum)
            or len(self.training_sample_ids) < 6
            or len(set(self.training_sample_ids)) != len(self.training_sample_ids)
            or any(not isinstance(value, str) or not value for value in self.training_sample_ids)
            or isinstance(self.training_trigger_minimum, bool)
            or isinstance(self.training_trigger_maximum, bool)
            or not self.minimum_trigger_policy_frame
            <= self.training_trigger_minimum
            <= self.parent_trigger_policy_frame
            <= self.training_trigger_maximum
            <= self.maximum_trigger_policy_frame
        ):
            raise ValueError("causal transition tensors or support are invalid")
        if (
            not math.isfinite(self.training_rmse_frames)
            or not 0.0 <= self.training_rmse_frames <= 20.0
            or isinstance(self.parent_trigger_policy_frame, bool)
            or not 75 <= self.parent_trigger_policy_frame <= 105
            or isinstance(self.maximum_trigger_residual_frames, bool)
            or not 2 <= self.maximum_trigger_residual_frames <= 20
            or not 70
            <= self.minimum_trigger_policy_frame
            < self.maximum_trigger_policy_frame
            <= 125
            or not 0.1 <= self.maximum_ood_distance <= 5.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.direct_joint_torque_output
            or self.schema_version != "rosclaw.growth.g1_causal_skill_transition_actor.v3"
        ):
            raise ValueError("causal transition actor violates its bounded SIM-only contract")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            **asdict(self),
            "feature_names": list(CAUSAL_TRANSITION_FEATURE_NAMES),
            "algorithm": "bounded_causal_transition_mlp_regression_v3",
            "serialized_executable_code": False,
            "pixels_used_for_training": False,
            "output_authority": "BOUNDED_SUCCESSOR_POLICY_PHASE_ONLY",
            "ood_route": "FROZEN_PARENT_TRIGGER",
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(self, features: NDArray[np.float64]) -> CausalTransitionDecision:
        observation = np.asarray(features, dtype=np.float64)
        if observation.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(observation)):
            raise ValueError("causal transition actor requires fourteen finite features")
        minimum = np.asarray(self.feature_minimum, dtype=np.float64)
        maximum = np.asarray(self.feature_maximum, dtype=np.float64)
        span = np.maximum(maximum - minimum, 0.05)
        support_distance = float(
            np.linalg.norm(
                (np.maximum(minimum - observation, 0.0) + np.maximum(observation - maximum, 0.0))
                / span
            )
        )
        normalized = (observation - np.asarray(self.feature_center)) / np.asarray(
            self.feature_scale
        )
        hidden = np.tanh(
            np.asarray(self.hidden_weights, dtype=np.float64) @ normalized
            + np.asarray(self.hidden_bias, dtype=np.float64)
        )
        raw = float(np.asarray(self.output_weights, dtype=np.float64) @ hidden + self.output_bias)
        residual = int(round(self.maximum_trigger_residual_frames * math.tanh(raw)))
        accepted = support_distance <= self.maximum_ood_distance
        trigger = self.parent_trigger_policy_frame
        if accepted:
            trigger = int(
                np.clip(
                    self.parent_trigger_policy_frame + residual,
                    self.training_trigger_minimum,
                    self.training_trigger_maximum,
                )
            )
            residual = trigger - self.parent_trigger_policy_frame
        else:
            residual = 0
        return CausalTransitionDecision(
            accepted=accepted,
            trigger_policy_frame=trigger,
            parent_trigger_policy_frame=self.parent_trigger_policy_frame,
            residual_frames=residual,
            support_distance=support_distance,
            actor_hash=self.actor_hash,
        )


def fit_causal_skill_transition_actor(
    samples: tuple[CausalTransitionSample, ...],
    *,
    source_stage_hash: str,
    seed: int = 1240,
    hidden_size: int = 16,
    epochs: int = 1200,
    learning_rate: float = 3.0e-3,
    weight_decay: float = 1.0e-4,
    parent_trigger_policy_frame: int = 88,
    maximum_trigger_residual_frames: int = 15,
    maximum_ood_distance: float = 2.0,
    device: str = "cpu",
) -> G1CausalSkillTransitionActor:
    """Fit one deterministic candidate from safe physics-selected labels."""

    safe = tuple(sample for sample in samples if sample.safe)
    if len(safe) != len(samples) or len(safe) < 6:
        raise ValueError("causal transition training requires six safe samples")
    if len({sample.sample_id for sample in safe}) != len(safe):
        raise ValueError("causal transition sample ids must be unique")
    if not source_stage_hash.startswith("sha256:") or len(source_stage_hash) != 71:
        raise ValueError("causal transition source stage must be content bound")
    if not 4 <= hidden_size <= 64 or not 50 <= epochs <= 10_000:
        raise ValueError("causal transition training shape or duration is invalid")
    observations = np.asarray([sample.features for sample in safe], dtype=np.float64)
    labels = np.asarray([sample.optimal_trigger_policy_frame for sample in safe], dtype=np.float64)
    center = np.mean(observations, axis=0)
    scale = np.std(observations, axis=0)
    scale = np.where(scale < 1.0e-6, 1.0, scale)
    normalized = (observations - center) / scale
    residual_targets = np.clip(
        (labels - parent_trigger_policy_frame) / maximum_trigger_residual_frames,
        -0.999,
        0.999,
    )

    import torch

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    training_device = torch.device(device)
    x = torch.as_tensor(normalized, dtype=torch.float32, device=training_device)
    y = torch.as_tensor(residual_targets[:, None], dtype=torch.float32, device=training_device)
    model = torch.nn.Sequential(
        torch.nn.Linear(_FEATURE_COUNT, hidden_size),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, 1),
        torch.nn.Tanh(),
    ).to(training_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for _ in range(epochs):
        prediction = model(x)
        loss = torch.mean((prediction - y) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    with torch.no_grad():
        prediction = model(x).detach().cpu().numpy()[:, 0]
    predicted_frames = parent_trigger_policy_frame + maximum_trigger_residual_frames * prediction
    rmse = float(np.sqrt(np.mean(np.square(predicted_frames - labels))))
    first = cast(Any, model[0])
    second = cast(Any, model[2])
    implementation_hash = hash_bytes(Path(__file__).read_bytes())
    snapshot_hash = hash_json(
        {
            "source_stage_hash": source_stage_hash,
            "implementation_hash": implementation_hash,
            "samples": [asdict(sample) for sample in safe],
            "seed": seed,
            "hidden_size": hidden_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "parent_trigger_policy_frame": parent_trigger_policy_frame,
            "maximum_trigger_residual_frames": maximum_trigger_residual_frames,
        }
    )
    output_weights = second.weight.detach().cpu().numpy().astype(np.float64)[0]
    output_bias = float(second.bias.detach().cpu().numpy().astype(np.float64)[0])
    return G1CausalSkillTransitionActor(
        source_stage_hash=source_stage_hash,
        training_snapshot_hash=str(snapshot_hash),
        implementation_hash=str(implementation_hash),
        feature_center=tuple(float(value) for value in center),
        feature_scale=tuple(float(value) for value in scale),
        feature_minimum=tuple(float(value) for value in np.min(observations, axis=0)),
        feature_maximum=tuple(float(value) for value in np.max(observations, axis=0)),
        hidden_weights=tuple(
            tuple(float(value) for value in row)
            for row in first.weight.detach().cpu().numpy().astype(np.float64)
        ),
        hidden_bias=tuple(
            float(value) for value in first.bias.detach().cpu().numpy().astype(np.float64)
        ),
        output_weights=tuple(float(value) for value in output_weights),
        output_bias=output_bias,
        training_sample_ids=tuple(sorted(sample.sample_id for sample in safe)),
        training_rmse_frames=rmse,
        training_trigger_minimum=int(np.min(labels)),
        training_trigger_maximum=int(np.max(labels)),
        parent_trigger_policy_frame=parent_trigger_policy_frame,
        maximum_trigger_residual_frames=maximum_trigger_residual_frames,
        maximum_ood_distance=maximum_ood_distance,
    )


def save_causal_skill_transition_actor(actor: G1CausalSkillTransitionActor, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_causal_skill_transition_actor(path: Path) -> CausalSkillTransitionActor:
    source = path.expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("causal transition actor artifact is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("causal transition actor artifact must be an object")
    if payload.get("schema_version") == "rosclaw.growth.causal_transition_risk_actor.v2":
        from rosclaw_soccer.growth.causal_skill_transition_risk import (
            load_causal_skill_transition_risk_actor,
        )

        return load_causal_skill_transition_risk_actor(source)
    if payload.get("schema_version") == "rosclaw.growth.causal_transition_memory_actor.v1":
        from rosclaw_soccer.growth.causal_skill_transition_risk import (
            load_causal_skill_transition_memory_actor,
        )

        return load_causal_skill_transition_memory_actor(source)
    claimed = payload.pop("actor_hash", None)
    if payload.pop("feature_names", None) != list(CAUSAL_TRANSITION_FEATURE_NAMES):
        raise ValueError("causal transition feature contract does not match")
    metadata = {
        "algorithm": "bounded_causal_transition_mlp_regression_v3",
        "serialized_executable_code": False,
        "pixels_used_for_training": False,
        "output_authority": "BOUNDED_SUCCESSOR_POLICY_PHASE_ONLY",
        "ood_route": "FROZEN_PARENT_TRIGGER",
    }
    if any(payload.pop(key, None) != value for key, value in metadata.items()):
        raise ValueError("causal transition metadata contract does not match")
    for key in (
        "feature_center",
        "feature_scale",
        "feature_minimum",
        "feature_maximum",
        "hidden_bias",
        "output_weights",
        "training_sample_ids",
    ):
        value = payload.get(key)
        if not isinstance(value, list):
            raise ValueError(f"causal transition actor {key} must be a list")
        payload[key] = tuple(value)
    hidden = payload.get("hidden_weights")
    if not isinstance(hidden, list):
        raise ValueError("causal transition hidden weights must be a matrix")
    payload["hidden_weights"] = tuple(tuple(row) for row in hidden)
    try:
        actor = G1CausalSkillTransitionActor(**payload)
    except TypeError as exc:
        raise ValueError("causal transition actor payload is invalid") from exc
    if claimed != actor.actor_hash:
        raise ValueError("causal transition actor hash does not match")
    return actor


def causal_transition_features(
    *,
    receiver_pelvis_world_m: NDArray[np.float64],
    predecessor_pelvis_world_m: NDArray[np.float64],
    receiver_ball_local_m: NDArray[np.float64],
    receiver_reception_target_local_m: NDArray[np.float64],
    receiver_shot_target_local_m: NDArray[np.float64],
    predecessor_swing_speed_scale: float,
    ball_ground_friction: float,
    predecessor_yaw_rad: float,
    receiver_kick_foot: str,
) -> NDArray[np.float64]:
    """Build the task/proprioceptive context shared by training and runtime."""

    receiver = np.asarray(receiver_pelvis_world_m, dtype=np.float64)
    predecessor = np.asarray(predecessor_pelvis_world_m, dtype=np.float64)
    ball = np.asarray(receiver_ball_local_m, dtype=np.float64)
    reception = np.asarray(receiver_reception_target_local_m, dtype=np.float64)
    shot = np.asarray(receiver_shot_target_local_m, dtype=np.float64)
    if (
        receiver.shape != (3,)
        or predecessor.shape != (3,)
        or ball.shape != (3,)
        or reception.shape != (3,)
        or shot.shape != (3,)
        or receiver_kick_foot not in {"left", "right"}
    ):
        raise ValueError("causal transition context shape is invalid")
    values = np.asarray(
        (
            *(predecessor - receiver)[:2],
            *ball,
            *reception[:2],
            shot[1],
            shot[2],
            predecessor_swing_speed_scale,
            ball_ground_friction,
            math.sin(predecessor_yaw_rad),
            math.cos(predecessor_yaw_rad),
            -1.0 if receiver_kick_foot == "left" else 1.0,
        ),
        dtype=np.float64,
    )
    if values.shape != (_FEATURE_COUNT,) or not np.all(np.isfinite(values)):
        raise ValueError("causal transition context must be finite")
    return values


__all__ = [
    "CAUSAL_TRANSITION_FEATURE_NAMES",
    "CausalSkillTransitionActor",
    "CausalTransitionDecision",
    "CausalTransitionSample",
    "G1CausalSkillTransitionActor",
    "causal_transition_features",
    "fit_causal_skill_transition_actor",
    "load_causal_skill_transition_actor",
    "save_causal_skill_transition_actor",
]
