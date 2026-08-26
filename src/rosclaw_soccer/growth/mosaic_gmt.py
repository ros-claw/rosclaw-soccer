"""Content-bound MOSAIC GMT cerebellar policy and overhead skill.

The public MOSAIC ``gmt.onnx`` policy is a closed-loop whole-body tracker.  It
must not be reduced to an open-loop pose clip: its observation history and
feedback action are what keep the robot upright while the arms move.  This
module validates that deployment contract, converts the small ONNX MLP to a
GPU-native Torch module, and binds a local motion skill to both the checkpoint
and its source data.

Everything in this module is ``SIM_ONLY``.  It has no ROS transport or motor
command path.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.mosaic_g1_contract import (
    MOSAIC_G1_ISAACLAB_BODY_NAMES,
    MOSAIC_G1_ISAACLAB_JOINT_NAMES,
    MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF,
    MOSAIC_G1_SEMANTIC_CONTRACT_HASH,
)
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_MAX_MODEL_BYTES = 64 * 1024 * 1024
_EXPECTED_OPS = (
    "Sub",
    "Div",
    "Gemm",
    "Elu",
    "Gemm",
    "Elu",
    "Gemm",
    "Elu",
    "Gemm",
    "Elu",
    "Gemm",
)
_EXPECTED_LAYER_SHAPES = ((1024, 770), (1024, 1024), (512, 1024), (256, 512), (29, 256))
_GMT_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


@dataclass(frozen=True)
class MosaicGMTContract:
    """Validated metadata and content identity of one GMT checkpoint."""

    checkpoint_hash: str
    topology_hash: str
    semantic_contract_hash: str
    raw_joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    anchor_body_name: str
    default_joint_position_rad: tuple[float, ...]
    joint_stiffness: tuple[float, ...]
    joint_damping: tuple[float, ...]
    action_scale: tuple[float, ...]
    observation_history_length: int
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.growth.mosaic_gmt_contract.v1"

    def __post_init__(self) -> None:
        arrays = (
            self.default_joint_position_rad,
            self.joint_stiffness,
            self.joint_damping,
            self.action_scale,
        )
        if (
            self.raw_joint_names != MOSAIC_G1_ISAACLAB_JOINT_NAMES
            or self.body_names != _GMT_BODY_NAMES
            or self.anchor_body_name != "torso_link"
            or any(len(value) != 29 for value in arrays)
            or any(not all(math.isfinite(item) for item in value) for value in arrays)
            or any(value <= 0.0 for value in self.joint_stiffness)
            or any(value <= 0.0 for value in self.joint_damping)
            or any(value <= 0.0 for value in self.action_scale)
            or self.observation_history_length != 5
            or self.semantic_contract_hash != MOSAIC_G1_SEMANTIC_CONTRACT_HASH
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("MOSAIC GMT metadata violates the closed-loop contract")

    @property
    def contract_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class G1MosaicGMTOverheadSkill:
    """A selected, independently qualified MOSAIC motion segment."""

    checkpoint_hash: str
    checkpoint_contract_hash: str
    source_hash: str
    qualification_hash: str
    semantic_contract_hash: str
    center_frame: int
    source_fps: float
    relative_times_sec: tuple[float, ...]
    raw_joint_position_rad: tuple[tuple[float, ...], ...]
    raw_joint_velocity_rad_s: tuple[tuple[float, ...], ...]
    aligned_torso_quaternion_wxyz: tuple[tuple[float, ...], ...]
    official_minimum_pelvis_height_m: float
    official_peak_minimum_bilateral_hand_height_m: float
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.growth.mosaic_gmt_overhead_skill.v1"

    def __post_init__(self) -> None:
        count = len(self.relative_times_sec)
        values = np.asarray(self.raw_joint_position_rad, dtype=np.float64)
        velocities = np.asarray(self.raw_joint_velocity_rad_s, dtype=np.float64)
        quaternions = np.asarray(self.aligned_torso_quaternion_wxyz, dtype=np.float64)
        if (
            count < 50
            or values.shape != (count, 29)
            or velocities.shape != values.shape
            or quaternions.shape != (count, 4)
            or not all(np.isfinite(value).all() for value in (values, velocities, quaternions))
            or not 30.0 <= self.source_fps <= 240.0
            or self.center_frame < 1
            or tuple(sorted(self.relative_times_sec)) != self.relative_times_sec
            or np.max(np.abs(np.linalg.norm(quaternions, axis=1) - 1.0)) > 2.0e-4
            or self.official_minimum_pelvis_height_m < 0.60
            or self.official_peak_minimum_bilateral_hand_height_m < 1.20
            or self.semantic_contract_hash != MOSAIC_G1_SEMANTIC_CONTRACT_HASH
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("MOSAIC GMT overhead skill is invalid or unqualified")

    @property
    def skill_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _metadata_array(metadata: dict[str, str], name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in metadata[name].split(","))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"MOSAIC GMT metadata `{name}` is invalid") from exc
    if len(values) != 29 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"MOSAIC GMT metadata `{name}` must contain 29 finite values")
    return values


def load_mosaic_gmt_torch(
    checkpoint_path: Path,
    *,
    device: Any,
) -> tuple[Any, MosaicGMTContract]:
    """Load a strictly validated ONNX MLP as a GPU-native Torch module."""

    path = checkpoint_path.expanduser().resolve()
    if not path.is_file() or not 0 < path.stat().st_size <= _MAX_MODEL_BYTES:
        raise ValueError("MOSAIC GMT checkpoint is missing or oversized")
    try:
        import onnx
        import torch
        from onnx import numpy_helper
    except ImportError as exc:  # pragma: no cover - optional training dependency
        raise RuntimeError(
            "MOSAIC GMT loading requires the optional onnx and torch packages"
        ) from exc

    document = onnx.load(str(path), load_external_data=False)
    if tuple(node.op_type for node in document.graph.node) != _EXPECTED_OPS:
        raise ValueError("MOSAIC GMT ONNX topology is not the audited feed-forward policy")
    if len(document.graph.input) != 1 or len(document.graph.output) != 1:
        raise ValueError("MOSAIC GMT ONNX input/output arity is invalid")
    input_shape = tuple(dim.dim_value for dim in document.graph.input[0].type.tensor_type.shape.dim)
    output_shape = tuple(
        dim.dim_value for dim in document.graph.output[0].type.tensor_type.shape.dim
    )
    if input_shape[-1:] != (770,) or output_shape[-1:] != (29,):
        raise ValueError("MOSAIC GMT ONNX tensor shape is invalid")
    initializers = {
        value.name: np.asarray(numpy_helper.to_array(value), dtype=np.float32)
        for value in document.graph.initializer
    }
    layer_names = ("actor.0", "actor.2", "actor.4", "actor.6", "actor.8")
    for name, shape in zip(layer_names, _EXPECTED_LAYER_SHAPES, strict=True):
        if initializers.get(name + ".weight", np.empty(0)).shape != shape:
            raise ValueError("MOSAIC GMT ONNX layer shape is invalid")
        if initializers.get(name + ".bias", np.empty(0)).shape != (shape[0],):
            raise ValueError("MOSAIC GMT ONNX bias shape is invalid")
    mean = initializers.get("normalizer._mean")
    divisor = initializers.get("onnx::Div_28")
    if (
        mean is None
        or divisor is None
        or mean.shape != (1, 770)
        or divisor.shape != (1, 770)
        or not np.isfinite(mean).all()
        or not np.isfinite(divisor).all()
        or np.any(divisor <= 0.0)
    ):
        raise ValueError("MOSAIC GMT observation normalizer is invalid")

    # ``dict(metadata_props)`` is not supported consistently across ONNX
    # versions; use the explicit protobuf fields.
    metadata = {item.key: item.value for item in document.metadata_props}
    raw_joint_names = tuple(value.strip() for value in metadata.get("joint_names", "").split(","))
    body_names = tuple(value.strip() for value in metadata.get("body_names", "").split(","))
    history = tuple(
        float(value)
        for value in metadata.get("observation_history_lengths", "").split(",")
        if value
    )
    if len(history) != 6 or any(value != 5.0 for value in history):
        raise ValueError("MOSAIC GMT observation history metadata is invalid")
    topology_hash = str(
        hash_json(
            {
                "ops": _EXPECTED_OPS,
                "layer_shapes": _EXPECTED_LAYER_SHAPES,
                "input": input_shape,
                "output": output_shape,
                "observation_names": metadata.get("observation_names"),
            }
        )
    )
    contract = MosaicGMTContract(
        checkpoint_hash=hash_bytes(path.read_bytes()),
        topology_hash=topology_hash,
        semantic_contract_hash=MOSAIC_G1_SEMANTIC_CONTRACT_HASH,
        raw_joint_names=raw_joint_names,
        body_names=body_names,
        anchor_body_name=metadata.get("anchor_body_name", ""),
        default_joint_position_rad=_metadata_array(metadata, "default_joint_pos"),
        joint_stiffness=_metadata_array(metadata, "joint_stiffness"),
        joint_damping=_metadata_array(metadata, "joint_damping"),
        action_scale=_metadata_array(metadata, "action_scale"),
        observation_history_length=5,
    )

    class _GMTPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("mean", torch.as_tensor(mean.copy()))  # type: ignore[union-attr]
            self.register_buffer(
                "divisor",
                torch.as_tensor(divisor.copy()),  # type: ignore[union-attr]
            )
            dimensions = (770, 1024, 1024, 512, 256, 29)
            self.layers = torch.nn.ModuleList(
                torch.nn.Linear(dimensions[index], dimensions[index + 1])
                for index in range(len(dimensions) - 1)
            )
            with torch.no_grad():
                for layer, name in zip(self.layers, layer_names, strict=True):
                    layer.weight.copy_(  # type: ignore[operator]
                        torch.as_tensor(initializers[name + ".weight"].copy())
                    )
                    layer.bias.copy_(  # type: ignore[operator]
                        torch.as_tensor(initializers[name + ".bias"].copy())
                    )

        def forward(self, observation: Any) -> Any:
            value = (observation - self.mean) / self.divisor
            for layer in self.layers[:-1]:
                value = torch.nn.functional.elu(layer(value))
            return self.layers[-1](value)

    policy = _GMTPolicy().to(device)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy, contract


def _quat_multiply_wxyz(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.float64]:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def build_g1_mosaic_gmt_overhead_skill(
    *,
    source_path: Path,
    checkpoint_path: Path,
    qualification_path: Path,
    output_path: Path,
    device: Any = "cpu",
) -> G1MosaicGMTOverheadSkill:
    """Select the strongest safe event from an independent official-physics exam."""

    source = source_path.expanduser().resolve()
    qualification = qualification_path.expanduser().resolve()
    if not source.is_file() or not qualification.is_file():
        raise ValueError("MOSAIC GMT skill sources are missing")
    _, contract = load_mosaic_gmt_torch(checkpoint_path, device=device)
    report = json.loads(qualification.read_text(encoding="utf-8"))
    if report.get("schema_version") != "rosclaw.mosaic_gmt.event_qualification.v1":
        raise ValueError("MOSAIC GMT event qualification schema is invalid")
    eligible = [
        event
        for event in report.get("events", [])
        if not event.get("failed", True)
        and int(event.get("completed_steps", 0)) >= 78
        and float(event.get("minimum_pelvis_height_m", 0.0)) >= 0.60
        and min(
            float(event.get("peak_left_hand_height_m", 0.0)),
            float(event.get("peak_right_hand_height_m", 0.0)),
        )
        >= 1.20
    ]
    if not eligible:
        raise ValueError("MOSAIC GMT has no independently qualified overhead event")
    selected = max(
        eligible,
        key=lambda event: min(
            float(event["peak_left_hand_height_m"]),
            float(event["peak_right_hand_height_m"]),
        ),
    )
    center = int(selected["center_frame"])
    with np.load(source, allow_pickle=False) as data:
        required = {"fps", "joint_pos", "joint_vel", "body_quat_w"}
        if not required.issubset(data.files):
            raise ValueError("MOSAIC GMT source arrays are incomplete")
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        raw_position = np.asarray(data["joint_pos"], dtype=np.float64)
        raw_velocity = np.asarray(data["joint_vel"], dtype=np.float64)
        raw_torso_quaternion = np.asarray(data["body_quat_w"][:, 9], dtype=np.float64)
    if (
        raw_position.ndim != 2
        or raw_position.shape[1] != 29
        or raw_velocity.shape != raw_position.shape
        or raw_torso_quaternion.shape != (raw_position.shape[0], 4)
        or not all(
            np.isfinite(value).all() for value in (raw_position, raw_velocity, raw_torso_quaternion)
        )
    ):
        raise ValueError("MOSAIC GMT source shapes or values are invalid")
    start = center - 28
    stop = center + 72
    if start < 0 or stop > raw_position.shape[0]:
        raise ValueError("MOSAIC GMT event window exceeds its source")
    initial = raw_torso_quaternion[0] / np.linalg.norm(raw_torso_quaternion[0])
    yaw = math.atan2(
        2.0 * (initial[0] * initial[3] + initial[1] * initial[2]),
        1.0 - 2.0 * (initial[2] ** 2 + initial[3] ** 2),
    )
    inverse_yaw = np.asarray((math.cos(yaw / 2.0), 0.0, 0.0, -math.sin(yaw / 2.0)))
    aligned = _quat_multiply_wxyz(
        np.broadcast_to(inverse_yaw, raw_torso_quaternion[start:stop].shape),
        raw_torso_quaternion[start:stop],
    )
    aligned /= np.linalg.norm(aligned, axis=1, keepdims=True)
    skill = G1MosaicGMTOverheadSkill(
        checkpoint_hash=contract.checkpoint_hash,
        checkpoint_contract_hash=contract.contract_hash,
        source_hash=hash_bytes(source.read_bytes()),
        qualification_hash=hash_bytes(qualification.read_bytes()),
        semantic_contract_hash=MOSAIC_G1_SEMANTIC_CONTRACT_HASH,
        center_frame=center,
        source_fps=fps,
        relative_times_sec=tuple(float((index - 28) / fps) for index in range(100)),
        raw_joint_position_rad=tuple(
            tuple(float(value) for value in row) for row in raw_position[start:stop]
        ),
        raw_joint_velocity_rad_s=tuple(
            tuple(float(value) for value in row) for row in raw_velocity[start:stop]
        ),
        aligned_torso_quaternion_wxyz=tuple(
            tuple(float(value) for value in row) for row in aligned
        ),
        official_minimum_pelvis_height_m=float(selected["minimum_pelvis_height_m"]),
        official_peak_minimum_bilateral_hand_height_m=min(
            float(selected["peak_left_hand_height_m"]),
            float(selected["peak_right_hand_height_m"]),
        ),
    )
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(skill_to_dict(skill), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    return skill


def skill_to_dict(skill: G1MosaicGMTOverheadSkill) -> dict[str, Any]:
    value = asdict(skill)
    value["joint_names"] = list(G1_DDS_JOINT_NAMES)
    value["raw_joint_names"] = list(MOSAIC_G1_ISAACLAB_JOINT_NAMES)
    value["raw_body_names"] = list(MOSAIC_G1_ISAACLAB_BODY_NAMES)
    value["skill_hash"] = skill.skill_hash
    return value


def load_g1_mosaic_gmt_overhead_skill(path: Path) -> G1MosaicGMTOverheadSkill:
    """Load and content-verify a JSON GMT skill without pickle."""

    source = path.expanduser().resolve()
    if not source.is_file() or source.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("MOSAIC GMT overhead skill is missing or oversized")
    payload = json.loads(source.read_text(encoding="utf-8"))
    expected_hash = payload.pop("skill_hash", None)
    if tuple(payload.pop("joint_names", ())) != G1_DDS_JOINT_NAMES:
        raise ValueError("MOSAIC GMT skill canonical joint names are invalid")
    if tuple(payload.pop("raw_joint_names", ())) != MOSAIC_G1_ISAACLAB_JOINT_NAMES:
        raise ValueError("MOSAIC GMT skill raw joint names are invalid")
    if tuple(payload.pop("raw_body_names", ())) != MOSAIC_G1_ISAACLAB_BODY_NAMES:
        raise ValueError("MOSAIC GMT skill raw body names are invalid")
    for name in (
        "relative_times_sec",
        "raw_joint_position_rad",
        "raw_joint_velocity_rad_s",
        "aligned_torso_quaternion_wxyz",
    ):
        payload[name] = tuple(
            tuple(float(item) for item in value) if isinstance(value, list) else float(value)
            for value in payload[name]
        )
    skill = G1MosaicGMTOverheadSkill(**payload)
    if expected_hash != skill.skill_hash:
        raise ValueError("MOSAIC GMT overhead skill content hash mismatch")
    return skill


class MosaicGMTTorchController:
    """Term-major five-frame history and action target for batched simulation."""

    def __init__(
        self,
        *,
        policy: Any,
        contract: MosaicGMTContract,
        skill: Any,
        environment_count: int,
        device: Any,
    ) -> None:
        import torch

        if (
            skill.checkpoint_hash != contract.checkpoint_hash
            or skill.checkpoint_contract_hash != contract.contract_hash
        ):
            raise ValueError("MOSAIC GMT skill/checkpoint binding is invalid")
        self.torch = torch
        self.policy = policy
        self.contract = contract
        self.skill = skill
        self.count = environment_count
        self.device = torch.device(device)
        self._widths = (58, 6, 3, 29, 29, 29)
        self._history = [
            torch.zeros((environment_count, 5, width), device=self.device) for width in self._widths
        ]
        self._previous_action = torch.zeros((environment_count, 29), device=self.device)
        self._times = torch.as_tensor(skill.relative_times_sec, device=self.device)
        self._position = torch.as_tensor(skill.raw_joint_position_rad, device=self.device)
        self._velocity = torch.as_tensor(skill.raw_joint_velocity_rad_s, device=self.device)
        self._quaternion = torch.as_tensor(skill.aligned_torso_quaternion_wxyz, device=self.device)
        self._default = torch.as_tensor(contract.default_joint_position_rad, device=self.device)
        self._scale = torch.as_tensor(contract.action_scale, device=self.device)
        self._canonical_from_raw = torch.as_tensor(
            MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF,
            dtype=torch.long,
            device=self.device,
        )
        inverse = np.empty(29, dtype=np.int64)
        inverse[np.asarray(MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF)] = np.arange(29)
        self._raw_from_canonical = torch.as_tensor(inverse, dtype=torch.long, device=self.device)

    def reset(self) -> None:
        for history in self._history:
            history.zero_()
        self._previous_action.zero_()

    def target(
        self,
        *,
        canonical_joint_position: Any,
        canonical_joint_velocity: Any,
        torso_quaternion_wxyz: Any,
        base_angular_velocity_body_rad_s: Any,
        heading_quaternion_wxyz: Any,
        relative_time_sec: Any,
        active: Any,
    ) -> tuple[Any, Any]:
        """Return canonical PD targets while updating only active histories."""

        torch = self.torch
        if tuple(canonical_joint_position.shape) != (self.count, 29):
            raise ValueError("MOSAIC GMT controller joint state shape is invalid")
        active = active.to(torch.bool)
        reference_position, reference_velocity, reference_quaternion = self._reference_at(
            relative_time_sec
        )
        world_reference = _quat_multiply_torch(heading_quaternion_wxyz, reference_quaternion)
        relative_quaternion = _quat_multiply_torch(
            _quat_inverse_torch(torso_quaternion_wxyz), world_reference
        )
        orientation = _quat_to_rotation_6d_torch(relative_quaternion)
        raw_position = canonical_joint_position[:, self._raw_from_canonical]
        raw_velocity = canonical_joint_velocity[:, self._raw_from_canonical]
        values = (
            torch.cat((reference_position, reference_velocity), dim=1),
            orientation,
            base_angular_velocity_body_rad_s,
            raw_position - self._default,
            raw_velocity,
            self._previous_action,
        )
        for history, value in zip(self._history, values, strict=True):
            shifted = torch.roll(history, shifts=-1, dims=1)
            shifted[:, -1] = value
            history.copy_(torch.where(active[:, None, None], shifted, torch.zeros_like(history)))
        observation = torch.cat([history.flatten(1) for history in self._history], dim=1)
        with torch.no_grad():
            action = self.policy(observation)
        action = torch.where(active[:, None], action, torch.zeros_like(action))
        self._previous_action.copy_(action)
        raw_target = self._default + self._scale * action
        return raw_target[:, self._canonical_from_raw], action

    def reference_target(self, relative_time_sec: Any) -> Any:
        """Return the canonical data trajectory for bounded feed-forward tracking."""

        position, _, _ = self._reference_at(relative_time_sec)
        return position[:, self._canonical_from_raw]

    def _reference_at(self, relative_time_sec: Any) -> tuple[Any, Any, Any]:
        torch = self.torch
        fractional = torch.clamp(
            (relative_time_sec - self._times[0]) / (self._times[1] - self._times[0]),
            0.0,
            float(self._times.shape[0] - 1),
        )
        lower = torch.floor(fractional).to(torch.long)
        upper = torch.clamp(lower + 1, max=self._times.shape[0] - 1)
        fraction = (fractional - lower.to(torch.float32)).unsqueeze(1)
        reference_position = (
            self._position[lower] * (1.0 - fraction) + self._position[upper] * fraction
        )
        reference_velocity = (
            self._velocity[lower] * (1.0 - fraction) + self._velocity[upper] * fraction
        )
        reference_quaternion = (
            self._quaternion[lower] * (1.0 - fraction) + self._quaternion[upper] * fraction
        )
        reference_quaternion /= torch.linalg.vector_norm(
            reference_quaternion, dim=1, keepdim=True
        ).clamp_min(1.0e-6)
        return reference_position, reference_velocity, reference_quaternion


def _quat_inverse_torch(quaternion: Any) -> Any:
    result = quaternion.clone()
    result[:, 1:] *= -1.0
    return result / quaternion.square().sum(dim=1, keepdim=True).clamp_min(1.0e-8)


def _quat_multiply_torch(left: Any, right: Any) -> Any:
    torch = __import__("torch")
    lw, lx, ly, lz = left.unbind(dim=1)
    rw, rx, ry, rz = right.unbind(dim=1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=1,
    )


def _quat_to_rotation_6d_torch(quaternion: Any) -> Any:
    torch = __import__("torch")
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=1, keepdim=True).clamp_min(
        1.0e-8
    )
    w, x, y, z = quaternion.unbind(dim=1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
        ),
        dim=1,
    )


__all__ = [
    "G1MosaicGMTOverheadSkill",
    "MosaicGMTContract",
    "MosaicGMTTorchController",
    "build_g1_mosaic_gmt_overhead_skill",
    "load_g1_mosaic_gmt_overhead_skill",
    "load_mosaic_gmt_torch",
    "skill_to_dict",
]
