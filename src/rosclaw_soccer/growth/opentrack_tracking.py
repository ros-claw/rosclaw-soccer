"""Content-bound Torch runtime for an OpenTrack full-body tracking expert."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_EXPECTED_OPS = (
    "MatMul",
    "Add",
    "Sigmoid",
    "Mul",
    "MatMul",
    "Add",
    "Sigmoid",
    "Mul",
    "MatMul",
    "Add",
    "Sigmoid",
    "Mul",
    "MatMul",
    "Add",
    "Sigmoid",
    "Mul",
    "MatMul",
    "Add",
    "Sigmoid",
    "Mul",
    "MatMul",
    "Add",
    "Split",
    "Tanh",
)

_WEIGHT_NAMES = tuple(
    f"mlp_1/hidden_{index}_1/Cast/ReadVariableOp:0" for index in range(6)
)
_BIAS_NAMES = tuple(
    f"mlp_1/hidden_{index}_1/BiasAdd/ReadVariableOp:0" for index in range(6)
)

# Exact OpenTrack G1TrackingGeneral v2 control contract.  These values are
# content-audited against the external config before the runtime is created.
OPENTRACK_DEFAULT_JOINT_POSITION = (
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.3,
    0.0,
    1.28,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.3,
    0.0,
    1.28,
    0.0,
    0.0,
    0.0,
)
OPENTRACK_JOINT_STIFFNESS = (
    100.0,
    100.0,
    100.0,
    200.0,
    80.0,
    20.0,
    100.0,
    100.0,
    100.0,
    200.0,
    80.0,
    20.0,
    300.0,
    300.0,
    300.0,
    90.0,
    60.0,
    20.0,
    60.0,
    20.0,
    20.0,
    20.0,
    90.0,
    60.0,
    20.0,
    60.0,
    20.0,
    20.0,
    20.0,
)
OPENTRACK_JOINT_DAMPING = (
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    1.0,
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    1.0,
    10.0,
    10.0,
    10.0,
    2.0,
    2.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    2.0,
    2.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
)


@dataclass(frozen=True)
class OpenTrackTrackingContract:
    policy_hash: str
    config_hash: str
    motion_hash: str
    observation_size: int = 156
    action_size: int = 29
    action_scale: float = 1.0
    control_dt_sec: float = 0.02
    joint_velocity_scale: float = 0.05
    difference_joint_velocity_scale: float = 0.05
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.opentrack_tracking_contract.v1"

    def __post_init__(self) -> None:
        hashes = (self.policy_hash, self.config_hash, self.motion_hash)
        if (
            any(not value.startswith("sha256:") or len(value) != 71 for value in hashes)
            or self.observation_size != 156
            or self.action_size != 29
            or not math.isclose(self.action_scale, 1.0, abs_tol=1.0e-12)
            or not math.isclose(self.control_dt_sec, 0.02, abs_tol=1.0e-12)
            or not math.isclose(self.joint_velocity_scale, 0.05, abs_tol=1.0e-12)
            or not math.isclose(
                self.difference_joint_velocity_scale, 0.05, abs_tol=1.0e-12
            )
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("OpenTrack tracking contract is invalid")

    @property
    def contract_hash(self) -> str:
        return str(hash_json(self.__dict__))


class OpenTrackTrackingTorchPolicy:
    """Exact six-layer SiLU actor exported by OpenTrack/Brax."""

    def __init__(self, *, weights: tuple[Any, ...], biases: tuple[Any, ...]) -> None:
        if len(weights) != 6 or len(biases) != 6:
            raise ValueError("OpenTrack Torch policy topology is invalid")
        self.weights = weights
        self.biases = biases

    def __call__(self, observation: Any) -> Any:
        import torch

        if observation.ndim != 2 or observation.shape[1] != 156:
            raise ValueError("OpenTrack Torch policy observation shape is invalid")
        hidden = observation
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            hidden = torch.nn.functional.silu(hidden @ weight + bias)
        output = hidden @ self.weights[-1] + self.biases[-1]
        return torch.tanh(output[:, :29])


def _validated_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    environment = payload.get("env_config") if isinstance(payload, dict) else None
    expected_keys = [
        "dif_joint_pos",
        "dif_joint_vel",
        "gvec_pelvis",
        "gyro_pelvis",
        "joint_pos",
        "joint_vel",
        "last_motor_targets",
        "ref_feet_height",
        "ref_root_height",
    ]
    if (
        not isinstance(environment, dict)
        or environment.get("obs_keys") != expected_keys
        or environment.get("history_len") != 0
        or not math.isclose(float(environment.get("ctrl_dt", -1.0)), 0.02)
        or not math.isclose(float(environment.get("action_scale", -1.0)), 1.0)
        or environment.get("obs_scales_config")
        != {"joint_vel": 0.05, "dif_joint_vel": 0.05}
    ):
        raise ValueError("OpenTrack tracking config is incompatible")
    return environment


def load_opentrack_tracking_torch(
    *,
    policy_path: Path,
    config_path: Path,
    motion_path: Path,
    device: Any,
) -> tuple[OpenTrackTrackingTorchPolicy, OpenTrackTrackingContract, dict[str, Any]]:
    """Load only allow-listed numeric assets into a Torch tracking runtime."""

    import onnx
    import torch
    from onnx import numpy_helper

    policy_file = policy_path.expanduser().resolve()
    config_file = config_path.expanduser().resolve()
    motion_file = motion_path.expanduser().resolve()
    if any(not path.is_file() for path in (policy_file, config_file, motion_file)):
        raise FileNotFoundError("OpenTrack tracking assets are incomplete")
    _validated_config(config_file)
    graph = onnx.load(str(policy_file))
    if (
        tuple(node.op_type for node in graph.graph.node) != _EXPECTED_OPS
        or tuple(item.name for item in graph.graph.input) != ("obs",)
        or tuple(item.name for item in graph.graph.output) != ("continuous_actions",)
    ):
        raise ValueError("OpenTrack ONNX graph topology is incompatible")
    initializers = {
        item.name: np.array(numpy_helper.to_array(item), copy=True)
        for item in graph.graph.initializer
    }
    if set(initializers) != set(_WEIGHT_NAMES + _BIAS_NAMES):
        raise ValueError("OpenTrack ONNX initializers are incompatible")
    expected_shapes = (
        ((156, 512), (512,)),
        ((512, 512), (512,)),
        ((512, 256), (256,)),
        ((256, 256), (256,)),
        ((256, 128), (128,)),
        ((128, 58), (58,)),
    )
    if any(
        initializers[weight].shape != weight_shape
        or initializers[bias].shape != bias_shape
        for weight, bias, (weight_shape, bias_shape) in zip(
            _WEIGHT_NAMES, _BIAS_NAMES, expected_shapes, strict=True
        )
    ):
        raise ValueError("OpenTrack ONNX parameter shape is incompatible")
    torch_device = torch.device(device)
    policy = OpenTrackTrackingTorchPolicy(
        weights=tuple(
            torch.as_tensor(initializers[name], device=torch_device) for name in _WEIGHT_NAMES
        ),
        biases=tuple(
            torch.as_tensor(initializers[name], device=torch_device) for name in _BIAS_NAMES
        ),
    )
    with np.load(motion_file, allow_pickle=False) as archive:
        required = {"qpos", "qvel", "site_xpos", "site_names", "frequency"}
        if not required.issubset(archive.files):
            raise ValueError("OpenTrack recovery motion archive is incomplete")
        qpos = np.array(archive["qpos"], copy=True)
        qvel = np.array(archive["qvel"], copy=True)
        site_xpos = np.array(archive["site_xpos"], copy=True)
        site_names = tuple(str(value) for value in archive["site_names"].tolist())
        frequency = float(archive["frequency"])
    if (
        qpos.ndim != 2
        or qpos.shape[1] != 36
        or qvel.shape != (qpos.shape[0], 35)
        or site_xpos.shape != (qpos.shape[0], len(site_names), 3)
        or not math.isclose(frequency, 50.0, abs_tol=1.0e-9)
        or any(not np.all(np.isfinite(value)) for value in (qpos, qvel, site_xpos))
    ):
        raise ValueError("OpenTrack recovery motion arrays are incompatible")
    feet_names = ("left_foot", "right_foot", "left_foot_top", "right_foot_top")
    try:
        feet_indices = tuple(site_names.index(name) for name in feet_names)
    except ValueError as exc:
        raise ValueError("OpenTrack recovery motion has no four-foot height contract") from exc
    references = {
        "qpos": torch.as_tensor(qpos, device=torch_device),
        "qvel": torch.as_tensor(qvel, device=torch_device),
        "feet_height": torch.as_tensor(
            site_xpos[:, feet_indices, 2], device=torch_device
        ),
    }
    contract = OpenTrackTrackingContract(
        policy_hash=hash_bytes(policy_file.read_bytes()),
        config_hash=hash_bytes(config_file.read_bytes()),
        motion_hash=hash_bytes(motion_file.read_bytes()),
    )
    return policy, contract, references


def _quat_rotate_inverse_torch(quaternion_wxyz: Any, vector: Any) -> Any:
    import torch

    quaternion = quaternion_wxyz / torch.linalg.vector_norm(
        quaternion_wxyz, dim=1, keepdim=True
    ).clamp_min(1.0e-8)
    w = quaternion[:, :1]
    xyz = quaternion[:, 1:]
    # Rotate by conjugate(q) without materializing a matrix.
    return vector - 2.0 * w * torch.cross(xyz, vector, dim=1) + 2.0 * torch.cross(
        xyz, torch.cross(xyz, vector, dim=1), dim=1
    )


def opentrack_tracking_observation_torch(
    *,
    canonical_joint_position: Any,
    canonical_joint_velocity: Any,
    pelvis_quaternion_wxyz: Any,
    root_angular_velocity_body_rad_s: Any,
    previous_motor_target: Any,
    reference_joint_position: Any,
    reference_joint_velocity: Any,
    reference_feet_height_m: Any,
    reference_root_height_m: Any,
) -> Any:
    """Build the exact 156-D OpenTrack actor observation on Torch tensors."""

    import torch

    count = int(canonical_joint_position.shape[0])
    expected = (
        (canonical_joint_position, (count, 29)),
        (canonical_joint_velocity, (count, 29)),
        (pelvis_quaternion_wxyz, (count, 4)),
        (root_angular_velocity_body_rad_s, (count, 3)),
        (previous_motor_target, (count, 29)),
        (reference_joint_position, (count, 29)),
        (reference_joint_velocity, (count, 29)),
        (reference_feet_height_m, (count, 4)),
        (reference_root_height_m, (count,)),
    )
    if any(tuple(value.shape) != shape for value, shape in expected):
        raise ValueError("OpenTrack tracking observation tensor shape is invalid")
    gravity_world = torch.zeros((count, 3), device=canonical_joint_position.device)
    gravity_world[:, 2] = -1.0
    gravity_body = _quat_rotate_inverse_torch(pelvis_quaternion_wxyz, gravity_world)
    # MuJoCo free-joint qvel stores the rotational component in the child
    # body frame.  OpenTrack's pelvis gyro sensor exposes the same local-frame
    # quantity, so rotating it again would corrupt the observation whenever
    # the fallen body is not upright.
    gyro_body = root_angular_velocity_body_rad_s
    default = torch.as_tensor(
        OPENTRACK_DEFAULT_JOINT_POSITION,
        dtype=canonical_joint_position.dtype,
        device=canonical_joint_position.device,
    )
    observation = torch.cat(
        (
            reference_joint_position - canonical_joint_position,
            0.05 * (reference_joint_velocity - canonical_joint_velocity),
            gravity_body,
            0.05 * gyro_body,
            canonical_joint_position - default,
            0.05 * canonical_joint_velocity,
            previous_motor_target,
            reference_feet_height_m,
            reference_root_height_m[:, None],
        ),
        dim=1,
    )
    if tuple(observation.shape) != (count, 156) or not bool(torch.isfinite(observation).all()):
        raise ValueError("OpenTrack tracking observation is non-finite")
    return observation


__all__ = [
    "OPENTRACK_DEFAULT_JOINT_POSITION",
    "OPENTRACK_JOINT_DAMPING",
    "OPENTRACK_JOINT_STIFFNESS",
    "OpenTrackTrackingContract",
    "OpenTrackTrackingTorchPolicy",
    "load_opentrack_tracking_torch",
    "opentrack_tracking_observation_torch",
]
