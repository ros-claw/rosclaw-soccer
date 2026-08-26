"""Content-bound Torch runtime for RoboNaldo's MuJoCo-native G1 get-up expert."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import g1_body_hash
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_EXPECTED_OPS = (
    "Cast",
    "Constant",
    "Squeeze",
    "Constant",
    "Clip",
    "Sub",
    "Div",
    "Gemm",
    "Elu",
    "Gemm",
    "Elu",
    "Gemm",
    "Elu",
    "Gemm",
    "Gather",
    "Gather",
    "Gather",
    "Gather",
    "Gather",
    "Gather",
)


@dataclass(frozen=True)
class MJLabGetUpContract:
    checkpoint_hash: str
    source_hash: str
    config_hash: str
    body_hash: str
    physics_scene_hash: str
    topology_hash: str
    joint_names: tuple[str, ...]
    joint_stiffness: tuple[float, ...]
    joint_damping: tuple[float, ...]
    default_joint_position_rad: tuple[float, ...]
    action_scale: tuple[float, ...]
    source_start_frame: int
    source_stop_frame: int
    source_fps: float
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.growth.mjlab_getup_contract.v1"

    def __post_init__(self) -> None:
        arrays = (
            self.joint_stiffness,
            self.joint_damping,
            self.default_joint_position_rad,
            self.action_scale,
        )
        hashes = (
            self.checkpoint_hash,
            self.source_hash,
            self.config_hash,
            self.body_hash,
            self.physics_scene_hash,
            self.topology_hash,
        )
        if (
            self.joint_names != G1_DDS_JOINT_NAMES
            or any(len(value) != 29 for value in arrays)
            or any(not all(math.isfinite(item) for item in value) for value in arrays)
            or any(value <= 0.0 for value in self.joint_stiffness)
            or any(value <= 0.0 for value in self.joint_damping)
            or any(value <= 0.0 for value in self.action_scale)
            or any(not value.startswith("sha256:") or len(value) != 71 for value in hashes)
            or self.source_start_frame != 500
            or self.source_stop_frame != 950
            or self.source_fps != 50.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("MJLab get-up contract is invalid")

    @property
    def contract_hash(self) -> str:
        return str(hash_json(self.__dict__))

    @property
    def duration_sec(self) -> float:
        return (self.source_stop_frame - self.source_start_frame - 1) / self.source_fps


@dataclass(frozen=True)
class MJLabRecoveryHandoffConfig:
    """Causal completion contract between a get-up expert and locomotion.

    A height crossing is deliberately insufficient.  The expert remains in
    closed loop at its terminal reference until both feet support the body and
    the root stays inside a low-momentum standing envelope for a continuous
    interval.  During that interval the downstream recurrent controller can
    observe the same causal state before it receives authority.
    """

    control_dt_sec: float = 0.02
    expert_time_scale: float = 1.0
    stable_hold_sec: float = 1.0
    blend_sec: float = 1.0
    minimum_pelvis_height_m: float = 0.72
    minimum_upright_projection: float = 0.95
    maximum_root_linear_speed_mps: float = 0.25
    maximum_root_angular_speed_rad_s: float = 0.50
    require_bilateral_foot_support: bool = True
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.growth.mjlab_recovery_handoff_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.control_dt_sec,
            self.expert_time_scale,
            self.stable_hold_sec,
            self.blend_sec,
            self.minimum_pelvis_height_m,
            self.minimum_upright_projection,
            self.maximum_root_linear_speed_mps,
            self.maximum_root_angular_speed_rad_s,
        )
        if (
            not all(math.isfinite(value) for value in values)
            or not 0.005 <= self.control_dt_sec <= 0.05
            or not 0.60 <= self.expert_time_scale <= 1.10
            or not 0.50 <= self.stable_hold_sec <= 3.0
            or not 0.25 <= self.blend_sec <= 2.0
            or not 0.65 <= self.minimum_pelvis_height_m <= 0.80
            or not 0.85 <= self.minimum_upright_projection <= 1.0
            or not 0.05 <= self.maximum_root_linear_speed_mps <= 0.50
            or not 0.10 <= self.maximum_root_angular_speed_rad_s <= 1.0
            or not isinstance(self.require_bilateral_foot_support, bool)
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("MJLab recovery handoff config is invalid")

    @property
    def stable_hold_steps(self) -> int:
        return int(math.ceil(self.stable_hold_sec / self.control_dt_sec))

    @property
    def config_hash(self) -> str:
        return str(hash_json(self.__dict__))


@dataclass(frozen=True)
class MJLabRecoveryHandoffSignals:
    stable_candidate: Any
    reset_locomotion: Any
    warm_locomotion: Any
    handoff_started: Any
    handoff_just_started: Any
    blend_fraction: Any


class MJLabRecoveryHandoff:
    """Batched, causal expert-to-locomotion transition state machine."""

    def __init__(
        self,
        *,
        config: MJLabRecoveryHandoffConfig,
        environment_count: int,
        device: Any,
    ) -> None:
        import torch

        if environment_count < 1:
            raise ValueError("MJLab recovery handoff requires at least one environment")
        self.torch = torch
        self.config = config
        self.count = environment_count
        self.device = torch.device(device)
        self._step = 0
        self._stable_streak = torch.zeros(environment_count, dtype=torch.long, device=self.device)
        self._handoff_step = torch.full_like(self._stable_streak, -1)

    def reset(self) -> None:
        self._step = 0
        self._stable_streak.zero_()
        self._handoff_step.fill_(-1)

    @property
    def stable_streak(self) -> Any:
        return self._stable_streak

    @property
    def handoff_step(self) -> Any:
        return self._handoff_step

    def update(
        self,
        *,
        pelvis_height_m: Any,
        upright_projection: Any,
        root_linear_speed_mps: Any,
        root_angular_speed_rad_s: Any,
        left_foot_supported: Any,
        right_foot_supported: Any,
        active: Any,
    ) -> MJLabRecoveryHandoffSignals:
        tensors = (
            pelvis_height_m,
            upright_projection,
            root_linear_speed_mps,
            root_angular_speed_rad_s,
            left_foot_supported,
            right_foot_supported,
            active,
        )
        if any(tuple(value.shape) != (self.count,) for value in tensors):
            raise ValueError("MJLab recovery handoff input shape is invalid")
        torch = self.torch
        finite = torch.isfinite(pelvis_height_m)
        finite &= torch.isfinite(upright_projection)
        finite &= torch.isfinite(root_linear_speed_mps)
        finite &= torch.isfinite(root_angular_speed_rad_s)
        stable = finite
        stable &= pelvis_height_m >= self.config.minimum_pelvis_height_m
        stable &= upright_projection >= self.config.minimum_upright_projection
        stable &= root_linear_speed_mps <= self.config.maximum_root_linear_speed_mps
        stable &= root_angular_speed_rad_s <= self.config.maximum_root_angular_speed_rad_s
        if self.config.require_bilateral_foot_support:
            stable &= left_foot_supported.to(torch.bool)
            stable &= right_foot_supported.to(torch.bool)
        stable &= active.to(torch.bool)
        not_started = self._handoff_step < 0
        reset_locomotion = stable & not_started & (self._stable_streak == 0)
        self._stable_streak.copy_(
            torch.where(
                stable & not_started,
                self._stable_streak + 1,
                torch.where(
                    not_started,
                    torch.zeros_like(self._stable_streak),
                    self._stable_streak,
                ),
            )
        )
        just_started = not_started & (self._stable_streak >= self.config.stable_hold_steps)
        self._handoff_step.copy_(
            torch.where(
                just_started,
                torch.full_like(self._handoff_step, self._step),
                self._handoff_step,
            )
        )
        started = self._handoff_step >= 0
        age_sec = torch.clamp(
            (torch.full_like(self._handoff_step, self._step) - self._handoff_step).to(torch.float32)
            * self.config.control_dt_sec,
            min=0.0,
        )
        blend = torch.where(
            started,
            torch.clamp(age_sec / self.config.blend_sec, 0.0, 1.0),
            torch.zeros(self.count, device=self.device),
        )
        signals = MJLabRecoveryHandoffSignals(
            stable_candidate=stable,
            reset_locomotion=reset_locomotion,
            warm_locomotion=(self._stable_streak > 0) | started,
            handoff_started=started,
            handoff_just_started=just_started,
            blend_fraction=blend,
        )
        self._step += 1
        return signals


def _metadata_vector(metadata: dict[str, str], name: str) -> tuple[float, ...]:
    try:
        value = tuple(float(item) for item in metadata[name].split(","))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"MJLab get-up metadata `{name}` is invalid") from exc
    if len(value) != 29 or not all(math.isfinite(item) for item in value):
        raise ValueError(f"MJLab get-up metadata `{name}` must contain 29 finite values")
    return value


def load_mjlab_getup_torch(
    *,
    checkpoint_path: Path,
    source_path: Path,
    config_path: Path,
    asset_root: Path,
    device: Any,
) -> tuple[Any, MJLabGetUpContract, dict[str, Any]]:
    """Validate the exported expert and convert only its MLP to frozen Torch."""

    import onnx
    import torch
    from onnx import numpy_helper
    from torch import nn

    checkpoint = checkpoint_path.expanduser().resolve()
    source = source_path.expanduser().resolve()
    config = config_path.expanduser().resolve()
    assets = asset_root.expanduser().resolve()
    scene = assets / "g1_description" / "scene_with_ball.xml"
    if (
        not checkpoint.is_file()
        or checkpoint.stat().st_size > 32 * 1024 * 1024
        or not source.is_file()
        or source.suffix != ".npz"
        or not config.is_file()
        or not scene.is_file()
    ):
        raise ValueError("MJLab get-up artifacts are missing or invalid")
    model = onnx.load(checkpoint, load_external_data=False)
    if tuple(node.op_type for node in model.graph.node) != _EXPECTED_OPS:
        raise ValueError("MJLab get-up ONNX topology is not allowlisted")
    metadata = {item.key: item.value for item in model.metadata_props}
    if (
        metadata.get("run_path") != "2026-03-10_11-44-32_g1_fallAndGetUp2_subject2_mj"
        or metadata.get("anchor_body_name") != "torso_link"
        or tuple(metadata.get("joint_names", "").split(",")) != G1_DDS_JOINT_NAMES
        or metadata.get("observation_names")
        != "command,motion_anchor_ori_b,base_ang_vel,joint_pos,joint_vel,actions"
    ):
        raise ValueError("MJLab get-up metadata semantics are invalid")
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    required = {
        "joint_pos.1",
        "joint_vel.1",
        "body_pos_w.1",
        "body_quat_w.1",
        "policy.obs_normalizer._mean",
        "onnx::Div_47",
        "policy.mlp.0.weight",
        "policy.mlp.0.bias",
        "policy.mlp.2.weight",
        "policy.mlp.2.bias",
        "policy.mlp.4.weight",
        "policy.mlp.4.bias",
        "policy.mlp.6.weight",
        "policy.mlp.6.bias",
    }
    if not required.issubset(initializers):
        raise ValueError("MJLab get-up ONNX tensors are incomplete")
    with np.load(source, allow_pickle=False) as data:
        source_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if not (
            np.array_equal(initializers["joint_pos.1"], data["joint_pos"])
            and np.array_equal(initializers["joint_vel.1"], data["joint_vel"])
        ):
            raise ValueError("MJLab get-up embedded command does not match its source NPZ")
    layers: list[Any] = []
    for index, (input_width, output_width) in enumerate(
        ((154, 512), (512, 256), (256, 128), (128, 29))
    ):
        source_index = 2 * index
        linear = nn.Linear(input_width, output_width)
        weight = initializers[f"policy.mlp.{source_index}.weight"]
        bias = initializers[f"policy.mlp.{source_index}.bias"]
        with torch.no_grad():
            linear.weight.copy_(torch.as_tensor(np.array(weight, copy=True)))
            linear.bias.copy_(torch.as_tensor(np.array(bias, copy=True)))
        layers.append(linear)
        if index < 3:
            layers.append(nn.ELU())

    class NormalizedPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(*layers)
            self.register_buffer(
                "mean",
                torch.as_tensor(np.array(initializers["policy.obs_normalizer._mean"], copy=True)),
            )
            self.register_buffer(
                "scale",
                torch.as_tensor(np.array(initializers["onnx::Div_47"], copy=True)),
            )

        def forward(self, observation: Any) -> Any:
            return self.network((observation - self.mean) / self.scale)

    policy = NormalizedPolicy().to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    topology = {
        "ops": list(_EXPECTED_OPS),
        "layer_shapes": [[154, 512], [512, 256], [256, 128], [128, 29]],
        "observation": metadata["observation_names"],
    }
    contract = MJLabGetUpContract(
        checkpoint_hash=hash_bytes(checkpoint.read_bytes()),
        source_hash=hash_bytes(source.read_bytes()),
        config_hash=hash_bytes(config.read_bytes()),
        body_hash=g1_body_hash(assets),
        physics_scene_hash=hash_bytes(scene.read_bytes()),
        topology_hash=hash_json(topology),
        joint_names=G1_DDS_JOINT_NAMES,
        joint_stiffness=_metadata_vector(metadata, "joint_stiffness"),
        joint_damping=_metadata_vector(metadata, "joint_damping"),
        default_joint_position_rad=_metadata_vector(metadata, "default_joint_pos"),
        action_scale=_metadata_vector(metadata, "action_scale"),
        source_start_frame=500,
        source_stop_frame=950,
        source_fps=source_fps,
    )
    start, stop = contract.source_start_frame, contract.source_stop_frame
    references = {
        "joint_position": torch.as_tensor(
            np.array(initializers["joint_pos.1"][start:stop], copy=True), device=device
        ),
        "joint_velocity": torch.as_tensor(
            np.array(initializers["joint_vel.1"][start:stop], copy=True), device=device
        ),
        "pelvis_position": torch.as_tensor(
            np.array(initializers["body_pos_w.1"][start:stop, 0], copy=True), device=device
        ),
        "pelvis_quaternion": torch.as_tensor(
            np.array(initializers["body_quat_w.1"][start:stop, 0], copy=True), device=device
        ),
        "torso_quaternion": torch.as_tensor(
            np.array(initializers["body_quat_w.1"][start:stop, 7], copy=True), device=device
        ),
    }
    return policy, contract, references


class MJLabGetUpTorchController:
    """Batched MuJoCo-order runtime matching the official 154-D deployment."""

    def __init__(
        self,
        *,
        policy: Any,
        contract: MJLabGetUpContract,
        references: dict[str, Any],
        environment_count: int,
        device: Any,
        warmup_steps: int = 10,
        initial_reference_frame: Any | None = None,
    ) -> None:
        import torch

        if warmup_steps != 10 or environment_count < 1:
            raise ValueError("MJLab get-up runtime dimensions are invalid")
        self.torch = torch
        self.policy = policy
        self.contract = contract
        self.references = references
        self.count = environment_count
        self.device = torch.device(device)
        self.warmup_steps = warmup_steps
        self._default = torch.as_tensor(contract.default_joint_position_rad, device=self.device)
        self._scale = torch.as_tensor(contract.action_scale, device=self.device)
        self._previous_action = torch.zeros((environment_count, 29), device=self.device)
        self._entry_position = torch.zeros_like(self._previous_action)
        self._heading = torch.zeros((environment_count, 4), device=self.device)
        self._heading[:, 0] = 1.0
        self._started = torch.zeros(environment_count, dtype=torch.bool, device=self.device)
        if initial_reference_frame is None:
            self._initial_reference_frame = torch.zeros(
                environment_count, dtype=torch.long, device=self.device
            )
        else:
            self._initial_reference_frame = torch.as_tensor(
                initial_reference_frame, dtype=torch.long, device=self.device
            ).clone()
        if (
            tuple(self._initial_reference_frame.shape) != (environment_count,)
            or bool((self._initial_reference_frame < 0).any())
            or bool(
                (self._initial_reference_frame >= self.references["joint_position"].shape[0]).any()
            )
        ):
            raise ValueError("MJLab get-up initial reference frame is invalid")

    @property
    def duration_sec(self) -> float:
        return self.contract.duration_sec + self.warmup_steps / self.contract.source_fps

    @property
    def initial_reference_frame(self) -> Any:
        return self._initial_reference_frame

    def set_initial_reference_frame_before_start(
        self,
        initial_reference_frame: Any,
        *,
        mask: Any | None = None,
    ) -> None:
        """Bind a causal entry phase before the selected worlds start.

        An impact absorber may deliberately hold the body before the get-up
        expert receives authority.  Phase selection must therefore happen at
        that boundary, not when the earlier high-momentum snapshot was taken.
        Once a world has entered the expert its reference phase is immutable.
        """

        torch = self.torch
        frame = torch.as_tensor(
            initial_reference_frame,
            dtype=torch.long,
            device=self.device,
        )
        if tuple(frame.shape) != (self.count,):
            raise ValueError("MJLab get-up initial reference frame shape is invalid")
        selected = (
            torch.ones(self.count, dtype=torch.bool, device=self.device)
            if mask is None
            else torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        )
        if tuple(selected.shape) != (self.count,):
            raise ValueError("MJLab get-up initial reference frame mask is invalid")
        if bool((self._started & selected).any()):
            raise RuntimeError("MJLab get-up reference phase is immutable after expert entry")
        selected_frame = frame[selected]
        if bool((selected_frame < 0).any()) or bool(
            (selected_frame >= self.references["joint_position"].shape[0]).any()
        ):
            raise ValueError("MJLab get-up initial reference frame is invalid")
        self._initial_reference_frame.copy_(
            torch.where(selected, frame, self._initial_reference_frame)
        )

    def reset(self) -> None:
        self._previous_action.zero_()
        self._entry_position.zero_()
        self._heading.zero_()
        self._heading[:, 0] = 1.0
        self._started.zero_()

    def target(
        self,
        *,
        canonical_joint_position: Any,
        canonical_joint_velocity: Any,
        torso_quaternion_wxyz: Any,
        base_angular_velocity_body_rad_s: Any,
        relative_time_sec: Any,
        active: Any,
    ) -> tuple[Any, Any]:
        torch = self.torch
        active = active.to(torch.bool)
        newly_active = active & ~self._started
        self._entry_position.copy_(
            torch.where(newly_active[:, None], canonical_joint_position, self._entry_position)
        )
        reference_initial = self.references["torso_quaternion"][self._initial_reference_frame]
        relative_initial = _quat_multiply_torch(
            torso_quaternion_wxyz,
            _quat_inverse_torch(reference_initial),
        )
        rw, rx, ry, rz = relative_initial.unbind(dim=1)
        yaw = torch.atan2(
            2.0 * (rw * rz + rx * ry),
            1.0 - 2.0 * (ry.square() + rz.square()),
        )
        new_heading = torch.stack(
            (
                torch.cos(0.5 * yaw),
                torch.zeros_like(yaw),
                torch.zeros_like(yaw),
                torch.sin(0.5 * yaw),
            ),
            dim=1,
        )
        self._heading.copy_(torch.where(newly_active[:, None], new_heading, self._heading))
        self._started |= active

        warmup_sec = self.warmup_steps / self.contract.source_fps
        policy_time = torch.clamp(relative_time_sec - warmup_sec, min=0.0)
        frame = torch.clamp(
            self._initial_reference_frame
            + torch.floor(policy_time * self.contract.source_fps).to(torch.long),
            max=self.references["joint_position"].shape[0] - 1,
        )
        reference_position = self.references["joint_position"][frame]
        reference_velocity = self.references["joint_velocity"][frame]
        reference_quaternion = self.references["torso_quaternion"][frame]
        world_reference = _quat_multiply_torch(self._heading, reference_quaternion)
        relative_quaternion = _quat_multiply_torch(
            _quat_inverse_torch(torso_quaternion_wxyz), world_reference
        )
        observation = torch.cat(
            (
                reference_position,
                reference_velocity,
                _quat_to_rotation_6d_torch(relative_quaternion),
                base_angular_velocity_body_rad_s,
                canonical_joint_position - self._default,
                canonical_joint_velocity,
                self._previous_action,
            ),
            dim=1,
        )
        with torch.no_grad():
            action = torch.clamp(self.policy(observation), -10.0, 10.0)
        policy_active = active & (relative_time_sec >= warmup_sec)
        action = torch.where(policy_active[:, None], action, torch.zeros_like(action))
        self._previous_action.copy_(action)
        policy_target = self._default + self._scale * action
        warmup_fraction = torch.clamp(relative_time_sec / warmup_sec, 0.0, 1.0)
        warmup_target = self._entry_position + warmup_fraction[:, None] * (
            self.references["joint_position"][self._initial_reference_frame] - self._entry_position
        )
        target = torch.where(policy_active[:, None], policy_target, warmup_target)
        return torch.where(active[:, None], target, canonical_joint_position), action


def estimate_mjlab_getup_reference_frame_torch(
    *,
    references: dict[str, Any],
    canonical_joint_position: Any,
    canonical_joint_velocity: Any,
    pelvis_height_m: Any,
    pelvis_quaternion_wxyz: Any,
) -> tuple[Any, dict[str, float]]:
    """Match each live body state to the closest demonstrated motion phase.

    The estimator is diagnostic and causal: it uses only proprioceptive state
    available at entry.  Quaternion distance is sign-invariant.  Fixed,
    reported weights keep the experiment auditable and avoid a learned router
    silently exploiting future trajectory state.
    """

    torch = __import__("torch")
    count = int(canonical_joint_position.shape[0])
    shapes = (
        tuple(canonical_joint_position.shape) == (count, 29),
        tuple(canonical_joint_velocity.shape) == (count, 29),
        tuple(pelvis_height_m.shape) == (count,),
        tuple(pelvis_quaternion_wxyz.shape) == (count, 4),
    )
    if not all(shapes):
        raise ValueError("MJLab get-up phase estimator input shape is invalid")
    weights = {
        "joint_position_mse": 1.0,
        "joint_velocity_mse": 0.05,
        "pelvis_height_squared_error": 2.0,
        "pelvis_quaternion_sign_invariant_error": 0.5,
    }
    position_error = (
        (references["joint_position"][None, :, :] - canonical_joint_position[:, None, :])
        .square()
        .mean(dim=2)
    )
    velocity_error = (
        (references["joint_velocity"][None, :, :] - canonical_joint_velocity[:, None, :])
        .square()
        .mean(dim=2)
    )
    height_error = (references["pelvis_position"][None, :, 2] - pelvis_height_m[:, None]).square()
    live_quaternion = pelvis_quaternion_wxyz / torch.linalg.vector_norm(
        pelvis_quaternion_wxyz, dim=1, keepdim=True
    ).clamp_min(1.0e-8)
    reference_quaternion = references["pelvis_quaternion"]
    reference_quaternion = reference_quaternion / torch.linalg.vector_norm(
        reference_quaternion, dim=1, keepdim=True
    ).clamp_min(1.0e-8)
    quaternion_dot = torch.einsum(
        "nfd,nfd->nf",
        live_quaternion[:, None, :].expand(-1, reference_quaternion.shape[0], -1),
        reference_quaternion[None, :, :].expand(count, -1, -1),
    ).abs()
    quaternion_error = 1.0 - quaternion_dot.square().clamp(max=1.0)
    cost = (
        weights["joint_position_mse"] * position_error
        + weights["joint_velocity_mse"] * velocity_error
        + weights["pelvis_height_squared_error"] * height_error
        + weights["pelvis_quaternion_sign_invariant_error"] * quaternion_error
    )
    finite = torch.isfinite(cost).all(dim=1)
    if not bool(finite.all()):
        raise ValueError("MJLab get-up phase estimator received non-finite state")
    return torch.argmin(cost, dim=1), weights


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
    "MJLabGetUpContract",
    "MJLabGetUpTorchController",
    "MJLabRecoveryHandoff",
    "MJLabRecoveryHandoffConfig",
    "MJLabRecoveryHandoffSignals",
    "estimate_mjlab_getup_reference_frame_torch",
    "load_mjlab_getup_torch",
]
