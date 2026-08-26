"""Content-bound task-space reach teacher for the G1 goalkeeper.

The online actor still owns the final residual action and physics PPO still
owns every return.  This module only converts a causal ball intercept into a
coherent shoulder/elbow/wrist warm-start target using the qualified G1 hand
Jacobians.  It never commands the simulator or hardware.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import g1_body_hash
from rosclaw_soccer.sim.contracts import hash_json

_ARM_RESIDUAL_LIMITS_RAD = np.asarray((0.70, 0.75, 0.55, 0.55, 0.18, 0.16, 0.16), dtype=np.float64)


@dataclass(frozen=True)
class GoalkeeperReachConfig:
    damping: float = 0.16
    reach_gain: float = 0.72
    maximum_position_error_m: float = 0.55
    primary_arm_scale: float = 1.0
    support_arm_scale: float = 0.42
    central_support_scale: float = 0.86
    arm_scale_transition_m: float = 0.35
    residual_scale: float = 0.70
    arm_authority_scale: float = 0.82
    workspace_scale: float = 1.0
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_reach_config.v2"

    def __post_init__(self) -> None:
        values = (
            self.damping,
            self.reach_gain,
            self.maximum_position_error_m,
            self.primary_arm_scale,
            self.support_arm_scale,
            self.central_support_scale,
            self.arm_scale_transition_m,
            self.residual_scale,
            self.arm_authority_scale,
            self.workspace_scale,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("goalkeeper reach settings must be finite and positive")
        if not 0.05 <= self.damping <= 0.40:
            raise ValueError("goalkeeper reach damping is outside [0.05, 0.40]")
        if not 0.25 <= self.reach_gain <= 1.0:
            raise ValueError("goalkeeper reach gain is outside [0.25, 1]")
        if not 0.25 <= self.maximum_position_error_m <= 0.80:
            raise ValueError("goalkeeper reach error bound is outside [0.25, 0.80] m")
        if not 0.10 <= self.support_arm_scale <= self.central_support_scale <= 1.0:
            raise ValueError("goalkeeper reach support-arm scales are invalid")
        if not 0.20 <= self.arm_scale_transition_m <= 0.60:
            raise ValueError("goalkeeper reach arm-scale transition is invalid")
        if not 0.40 <= self.residual_scale <= 1.0:
            raise ValueError("goalkeeper reach residual scale is invalid")
        if not 0.40 <= self.arm_authority_scale <= 0.90:
            raise ValueError("goalkeeper reach arm authority is invalid")
        if not 0.50 <= self.workspace_scale <= 2.50:
            raise ValueError("goalkeeper reach workspace scale is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("goalkeeper reach teacher is SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class G1TaskSpaceReachModel:
    body_hash: str
    config_hash: str
    left_damped_inverse: tuple[tuple[float, ...], ...]
    right_damped_inverse: tuple[tuple[float, ...], ...]
    left_ready_hand_relative_m: tuple[float, float, float]
    right_ready_hand_relative_m: tuple[float, float, float]
    effective_arm_limits_rad: tuple[float, ...]
    reach_gain: float
    maximum_position_error_m: float
    primary_arm_scale: float
    support_arm_scale: float
    central_support_scale: float
    arm_scale_transition_m: float
    joint_authority: str = "SHOULDERS_ELBOWS_WRISTS_TEACHER_ONLY"
    physics_authority: bool = False
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.g1_task_space_reach_model.v2"

    def __post_init__(self) -> None:
        if not self.body_hash.startswith("sha256:") or not self.config_hash.startswith("sha256:"):
            raise ValueError("goalkeeper reach model requires content hashes")
        for matrix in (self.left_damped_inverse, self.right_damped_inverse):
            array = np.asarray(matrix, dtype=np.float64)
            if array.shape != (7, 3) or not np.all(np.isfinite(array)):
                raise ValueError("goalkeeper reach inverse must have finite shape (7, 3)")
        if len(self.effective_arm_limits_rad) != 7 or any(
            not math.isfinite(value) or value <= 0.0 for value in self.effective_arm_limits_rad
        ):
            raise ValueError("goalkeeper reach effective arm limits are invalid")
        settings = (
            self.reach_gain,
            self.maximum_position_error_m,
            self.primary_arm_scale,
            self.support_arm_scale,
            self.central_support_scale,
            self.arm_scale_transition_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in settings):
            raise ValueError("goalkeeper reach model settings are invalid")
        if self.physics_authority or self.activation_ceiling != "SIM_ONLY":
            raise ValueError("goalkeeper reach model exceeds its authority")

    @property
    def model_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class GoalkeeperReachAtlasConfig:
    """Bounded nonlinear IK curriculum used to manufacture teacher labels."""

    iterations: int = 64
    damping: float = 0.10
    step_limit_rad: float = 0.10
    error_tolerance_m: float = 0.012
    target_x_m: float = -0.08
    lateral_targets_m: tuple[float, ...] = (-1.05, -0.70, -0.35, 0.0, 0.35, 0.70, 1.05)
    relative_height_targets_m: tuple[float, ...] = (-0.62, -0.32, -0.02, 0.28, 0.58, 0.82)
    interpolation_distance_scales_m: tuple[float, float, float] = (0.30, 0.30, 0.24)
    interpolation_neighbors: int = 4
    interpolation_kernel: Literal["inverse_distance", "gaussian"] = "inverse_distance"
    interpolation_temperature: float = 0.75
    multistart_count: int = 1
    multistart_seed: int = 25_101
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_reach_atlas_config.v3"

    def __post_init__(self) -> None:
        if not 8 <= self.iterations <= 256:
            raise ValueError("goalkeeper reach atlas iteration count is invalid")
        values = (self.damping, self.step_limit_rad, self.error_tolerance_m)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("goalkeeper reach atlas solver settings must be finite and positive")
        if not 0.04 <= self.damping <= 0.30:
            raise ValueError("goalkeeper reach atlas damping is invalid")
        if not 0.02 <= self.step_limit_rad <= 0.20:
            raise ValueError("goalkeeper reach atlas step limit is invalid")
        if not 0.002 <= self.error_tolerance_m <= 0.05:
            raise ValueError("goalkeeper reach atlas error tolerance is invalid")
        if not math.isfinite(self.target_x_m) or not -0.40 <= self.target_x_m <= 0.10:
            raise ValueError("goalkeeper reach atlas x target is invalid")
        lateral = np.asarray(self.lateral_targets_m, dtype=np.float64)
        heights = np.asarray(self.relative_height_targets_m, dtype=np.float64)
        if (
            lateral.ndim != 1
            or heights.ndim != 1
            or lateral.size < 5
            or heights.size < 4
            or not np.all(np.isfinite(lateral))
            or not np.all(np.isfinite(heights))
            or not np.all(np.diff(lateral) > 0.0)
            or not np.all(np.diff(heights) > 0.0)
            or not np.allclose(lateral, -lateral[::-1], atol=1.0e-9, rtol=0.0)
        ):
            raise ValueError("goalkeeper reach atlas grid is invalid")
        scales = np.asarray(self.interpolation_distance_scales_m, dtype=np.float64)
        if scales.shape != (3,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("goalkeeper reach atlas interpolation scales are invalid")
        anchor_count = int(lateral.size * heights.size)
        if not 1 <= self.interpolation_neighbors <= anchor_count:
            raise ValueError("goalkeeper reach atlas neighbor count is invalid")
        if self.interpolation_kernel not in {"inverse_distance", "gaussian"} or not (
            math.isfinite(self.interpolation_temperature)
            and 0.10 <= self.interpolation_temperature <= 4.0
        ):
            raise ValueError("goalkeeper reach atlas interpolation kernel is invalid")
        if not 1 <= self.multistart_count <= 32 or not 0 <= self.multistart_seed < 2**31:
            raise ValueError("goalkeeper reach atlas multistart settings are invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("goalkeeper reach atlas is SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class G1TaskSpaceReachAtlas:
    """Content-bound nonlinear hand targets over the qualified G1 workspace."""

    body_hash: str
    config_hash: str
    atlas_config_hash: str
    target_relative_m: tuple[tuple[float, float, float], ...]
    left_normalized_action: tuple[tuple[float, ...], ...]
    right_normalized_action: tuple[tuple[float, ...], ...]
    left_terminal_error_m: tuple[float, ...]
    right_terminal_error_m: tuple[float, ...]
    effective_arm_limits_rad: tuple[float, ...]
    interpolation_distance_scales_m: tuple[float, float, float]
    interpolation_neighbors: int
    interpolation_kernel: Literal["inverse_distance", "gaussian"]
    interpolation_temperature: float
    reach_gain: float
    primary_arm_scale: float
    support_arm_scale: float
    central_support_scale: float
    arm_scale_transition_m: float
    interpolation_symmetry: str = "POSITIVE_HALF_SPACE_EXACT_ARM_REFLECTION_V1"
    joint_authority: str = "SHOULDERS_ELBOWS_WRISTS_TEACHER_ONLY"
    solver: str = "DETERMINISTIC_MULTISTART_JOINT_BOUNDED_DAMPED_LEAST_SQUARES"
    physics_authority: bool = False
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.g1_task_space_reach_atlas.v5"

    def __post_init__(self) -> None:
        hashes = (self.body_hash, self.config_hash, self.atlas_config_hash)
        if any(not value.startswith("sha256:") for value in hashes):
            raise ValueError("goalkeeper reach atlas requires content hashes")
        targets = np.asarray(self.target_relative_m, dtype=np.float64)
        left = np.asarray(self.left_normalized_action, dtype=np.float64)
        right = np.asarray(self.right_normalized_action, dtype=np.float64)
        if (
            targets.ndim != 2
            or targets.shape[1] != 3
            or targets.shape[0] < 20
            or left.shape != (targets.shape[0], 7)
            or right.shape != left.shape
            or not np.all(np.isfinite(targets))
            or not np.all(np.isfinite(left))
            or not np.all(np.isfinite(right))
            or np.max(np.abs(left)) > 1.0 + 1.0e-9
            or np.max(np.abs(right)) > 1.0 + 1.0e-9
        ):
            raise ValueError("goalkeeper reach atlas tensors are invalid")
        errors = np.asarray(
            (self.left_terminal_error_m, self.right_terminal_error_m), dtype=np.float64
        )
        if errors.shape != (2, targets.shape[0]) or not np.all(np.isfinite(errors)):
            raise ValueError("goalkeeper reach atlas errors are invalid")
        if np.any(errors < 0.0) or np.any(errors > 3.0):
            raise ValueError("goalkeeper reach atlas errors exceed their physical envelope")
        limits = np.asarray(self.effective_arm_limits_rad, dtype=np.float64)
        scales = np.asarray(self.interpolation_distance_scales_m, dtype=np.float64)
        if limits.shape != (7,) or np.any(limits <= 0.0) or not np.all(np.isfinite(limits)):
            raise ValueError("goalkeeper reach atlas joint limits are invalid")
        if scales.shape != (3,) or np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
            raise ValueError("goalkeeper reach atlas interpolation scales are invalid")
        if not 1 <= self.interpolation_neighbors <= targets.shape[0]:
            raise ValueError("goalkeeper reach atlas neighbor count is invalid")
        if self.interpolation_kernel not in {"inverse_distance", "gaussian"} or not (
            math.isfinite(self.interpolation_temperature)
            and 0.10 <= self.interpolation_temperature <= 4.0
        ):
            raise ValueError("goalkeeper reach atlas interpolation kernel is invalid")
        if self.interpolation_symmetry != "POSITIVE_HALF_SPACE_EXACT_ARM_REFLECTION_V1":
            raise ValueError("goalkeeper reach atlas symmetry contract is invalid")
        settings = (
            self.reach_gain,
            self.primary_arm_scale,
            self.support_arm_scale,
            self.central_support_scale,
            self.arm_scale_transition_m,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in settings):
            raise ValueError("goalkeeper reach atlas scales are invalid")
        if self.physics_authority or self.activation_ceiling != "SIM_ONLY":
            raise ValueError("goalkeeper reach atlas exceeds its authority")

    @property
    def model_hash(self) -> str:
        return str(hash_json(asdict(self)))


def build_g1_task_space_reach_model(
    asset_root: Path,
    *,
    config: GoalkeeperReachConfig | None = None,
) -> G1TaskSpaceReachModel:
    """Build ready-pose hand Jacobian inverses from the exact G1 Body."""

    import mujoco

    from rosclaw_soccer.training.goalkeeper_mjwarp import _LOCO_DEFAULT, _LOCO_TO_MOTOR
    from rosclaw_soccer.world.field import build_g1_stadium_model

    active = config or GoalkeeperReachConfig()
    root = asset_root.expanduser().resolve()
    model = build_g1_stadium_model(root)
    data = mujoco.MjData(model)
    ready = np.zeros(29, dtype=np.float64)
    ready[np.asarray(_LOCO_TO_MOTOR, dtype=np.int64)] = np.asarray(_LOCO_DEFAULT, dtype=np.float64)
    data.qpos[:7] = (4.52, 0.0, 0.793, 0.0, 0.0, 0.0, 1.0)
    data.qpos[7:36] = ready
    mujoco.mj_forward(model, data)

    def one(side: str, motor_indices: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        geom = int(model.geom(f"{side}_hand_collision").id)
        jacobian = np.zeros((3, model.nv), dtype=np.float64)
        angular = np.zeros_like(jacobian)
        mujoco.mj_jacGeom(model, data, jacobian, angular, geom)
        arm = jacobian[:, np.asarray(tuple(6 + index for index in motor_indices))]
        damped = arm.T @ np.linalg.inv(arm @ arm.T + active.damping * active.damping * np.eye(3))
        relative = np.asarray(data.geom_xpos[geom] - data.qpos[:3], dtype=np.float64)
        return damped, relative

    left, left_relative = one("left", tuple(range(15, 22)))
    right, right_relative = one("right", tuple(range(22, 29)))
    limits = (
        _ARM_RESIDUAL_LIMITS_RAD
        * active.residual_scale
        * active.arm_authority_scale
        * active.workspace_scale
    )
    return G1TaskSpaceReachModel(
        body_hash=g1_body_hash(root),
        config_hash=active.config_hash,
        left_damped_inverse=tuple(tuple(float(value) for value in row) for row in left),
        right_damped_inverse=tuple(tuple(float(value) for value in row) for row in right),
        left_ready_hand_relative_m=(
            float(left_relative[0]),
            float(left_relative[1]),
            float(left_relative[2]),
        ),
        right_ready_hand_relative_m=(
            float(right_relative[0]),
            float(right_relative[1]),
            float(right_relative[2]),
        ),
        effective_arm_limits_rad=tuple(float(value) for value in limits),
        reach_gain=active.reach_gain,
        maximum_position_error_m=active.maximum_position_error_m,
        primary_arm_scale=active.primary_arm_scale,
        support_arm_scale=active.support_arm_scale,
        central_support_scale=active.central_support_scale,
        arm_scale_transition_m=active.arm_scale_transition_m,
    )


def build_g1_task_space_reach_atlas(
    asset_root: Path,
    *,
    config: GoalkeeperReachConfig | None = None,
    atlas_config: GoalkeeperReachAtlasConfig | None = None,
) -> G1TaskSpaceReachAtlas:
    """Solve a bounded nonlinear arm workspace from the exact qualified Body."""

    import mujoco

    from rosclaw_soccer.training.goalkeeper_mjwarp import _LOCO_DEFAULT, _LOCO_TO_MOTOR
    from rosclaw_soccer.world.field import build_g1_stadium_model

    reach = config or GoalkeeperReachConfig()
    atlas = atlas_config or GoalkeeperReachAtlasConfig()
    root = asset_root.expanduser().resolve()
    mj_model = build_g1_stadium_model(root)
    data = mujoco.MjData(mj_model)
    ready = np.zeros(29, dtype=np.float64)
    ready[np.asarray(_LOCO_TO_MOTOR, dtype=np.int64)] = np.asarray(_LOCO_DEFAULT, dtype=np.float64)
    limits = (
        _ARM_RESIDUAL_LIMITS_RAD
        * reach.residual_scale
        * reach.arm_authority_scale
        * reach.workspace_scale
    )
    targets = tuple(
        (atlas.target_x_m, float(y), float(z))
        for y in atlas.lateral_targets_m
        for z in atlas.relative_height_targets_m
    )

    def solve(
        side: str,
        motor_indices: tuple[int, ...],
        target: tuple[float, float, float],
        target_index: int,
    ) -> tuple[tuple[float, ...], float]:
        qpos_indices = np.asarray(tuple(7 + index for index in motor_indices), dtype=np.int64)
        dof_indices = np.asarray(tuple(6 + index for index in motor_indices), dtype=np.int64)
        lower = ready[np.asarray(motor_indices, dtype=np.int64)] - limits
        upper = ready[np.asarray(motor_indices, dtype=np.int64)] + limits
        for offset, qpos_index in enumerate(qpos_indices):
            joint_candidates = np.flatnonzero(mj_model.jnt_qposadr == qpos_index)
            if joint_candidates.size != 1:
                raise RuntimeError("qualified G1 arm joint mapping changed")
            joint = int(joint_candidates[0])
            if bool(mj_model.jnt_limited[joint]):
                lower[offset] = max(lower[offset], float(mj_model.jnt_range[joint, 0]))
                upper[offset] = min(upper[offset], float(mj_model.jnt_range[joint, 1]))
        target_array = np.asarray(target, dtype=np.float64)
        geom = int(mj_model.geom(f"{side}_hand_collision").id)
        side_offset = 0 if side == "left" else 1_000_003
        rng = np.random.default_rng(atlas.multistart_seed + side_offset + 7_919 * target_index)
        best_error = math.inf
        best_position = ready[np.asarray(motor_indices, dtype=np.int64)].copy()
        for restart in range(atlas.multistart_count):
            data.qpos.fill(0.0)
            data.qvel.fill(0.0)
            data.qpos[:7] = (4.52, 0.0, 0.793, 0.0, 0.0, 0.0, 1.0)
            data.qpos[7:36] = ready
            if restart > 0:
                data.qpos[qpos_indices] = rng.uniform(lower, upper)
            for _ in range(atlas.iterations):
                mujoco.mj_forward(mj_model, data)
                relative = np.asarray(data.geom_xpos[geom] - data.qpos[:3], dtype=np.float64)
                error = target_array - relative
                if float(np.linalg.norm(error)) <= atlas.error_tolerance_m:
                    break
                jacobian = np.zeros((3, mj_model.nv), dtype=np.float64)
                angular = np.zeros_like(jacobian)
                mujoco.mj_jacGeom(mj_model, data, jacobian, angular, geom)
                arm = jacobian[:, dof_indices]
                damped = arm.T @ np.linalg.inv(
                    arm @ arm.T + atlas.damping * atlas.damping * np.eye(3)
                )
                delta = np.clip(
                    damped @ error,
                    -atlas.step_limit_rad,
                    atlas.step_limit_rad,
                )
                data.qpos[qpos_indices] = np.clip(data.qpos[qpos_indices] + delta, lower, upper)
            mujoco.mj_forward(mj_model, data)
            terminal = np.asarray(data.geom_xpos[geom] - data.qpos[:3], dtype=np.float64)
            terminal_error = float(np.linalg.norm(target_array - terminal))
            if terminal_error < best_error:
                best_error = terminal_error
                best_position = data.qpos[qpos_indices].copy()
        normalized = np.clip(
            (best_position - ready[np.asarray(motor_indices, dtype=np.int64)]) / limits,
            -1.0,
            1.0,
        )
        return (
            tuple(float(value) for value in normalized),
            best_error,
        )

    left_solutions = [
        solve("left", tuple(range(15, 22)), target, index) for index, target in enumerate(targets)
    ]
    right_solutions = [
        solve("right", tuple(range(22, 29)), target, index) for index, target in enumerate(targets)
    ]
    return G1TaskSpaceReachAtlas(
        body_hash=g1_body_hash(root),
        config_hash=reach.config_hash,
        atlas_config_hash=atlas.config_hash,
        target_relative_m=targets,
        left_normalized_action=tuple(solution[0] for solution in left_solutions),
        right_normalized_action=tuple(solution[0] for solution in right_solutions),
        left_terminal_error_m=tuple(solution[1] for solution in left_solutions),
        right_terminal_error_m=tuple(solution[1] for solution in right_solutions),
        effective_arm_limits_rad=tuple(float(value) for value in limits),
        interpolation_distance_scales_m=atlas.interpolation_distance_scales_m,
        interpolation_neighbors=atlas.interpolation_neighbors,
        interpolation_kernel=atlas.interpolation_kernel,
        interpolation_temperature=atlas.interpolation_temperature,
        reach_gain=reach.reach_gain,
        primary_arm_scale=reach.primary_arm_scale,
        support_arm_scale=reach.support_arm_scale,
        central_support_scale=reach.central_support_scale,
        arm_scale_transition_m=reach.arm_scale_transition_m,
    )


def task_space_reach_teacher_action(
    observation: Any,
    *,
    model: G1TaskSpaceReachModel | G1TaskSpaceReachAtlas,
    config: GoalkeeperReachConfig | None = None,
    allow_intent_cue: bool = False,
) -> Any:
    """Map causal intercept observations to bounded coherent arm targets.

    A 77-D observation may optionally activate the reach while the ball is
    still attached to the shooter, but only when the public noisy intent cue
    is visible.  The 74-D contract and dropped cues retain the historical
    shot-in-flight-only behaviour.
    """

    import torch

    if config is not None and model.config_hash != config.config_hash:
        raise ValueError("goalkeeper reach model/config mismatch")
    if observation.ndim != 2 or observation.shape[1] not in {74, 77}:
        raise ValueError("goalkeeper task-space teacher expects batched 74-D or 77-D observations")
    if not bool(torch.all(torch.isfinite(observation))):
        raise ValueError("goalkeeper task-space observation must be finite")
    target = observation[:, 6:9] * 2.0
    shot_active = observation[:, -3] + observation[:, -2] >= 0.5
    predictive_ready = torch.zeros_like(shot_active)
    if allow_intent_cue and observation.shape[1] == 77:
        predictive_ready = (observation[:, -4] > 0.5) & (observation[:, -5] > 0.5)
    target_y = target[:, 1]
    side_weight = torch.clamp(torch.abs(target_y) / model.arm_scale_transition_m, 0.0, 1.0)
    side_weight = side_weight * side_weight * (3.0 - 2.0 * side_weight)
    primary_left = target_y < 0.0
    output = torch.zeros((observation.shape[0], 18), device=observation.device)
    limits = torch.as_tensor(
        model.effective_arm_limits_rad,
        dtype=observation.dtype,
        device=observation.device,
    )

    def solve(
        *,
        inverse: tuple[tuple[float, ...], ...],
        ready: tuple[float, float, float],
        left: bool,
    ) -> Any:
        if not isinstance(model, G1TaskSpaceReachModel):
            raise RuntimeError("linear reach solver received nonlinear atlas")
        pinv = torch.as_tensor(inverse, dtype=observation.dtype, device=observation.device)
        hand = torch.as_tensor(ready, dtype=observation.dtype, device=observation.device)
        error = target - hand.unsqueeze(0)
        norm = torch.linalg.vector_norm(error, dim=1, keepdim=True)
        error *= torch.clamp(model.maximum_position_error_m / (norm + 1.0e-8), max=1.0)
        joint = error @ pinv.T
        normalized = model.reach_gain * joint / limits.unsqueeze(0)
        primary = primary_left if left else ~primary_left
        side_scale = torch.where(
            primary,
            torch.full_like(target_y, model.primary_arm_scale),
            torch.full_like(target_y, model.support_arm_scale),
        )
        scale = model.central_support_scale + side_weight * (
            side_scale - model.central_support_scale
        )
        return torch.clamp(normalized * scale.unsqueeze(1), -1.0, 1.0)

    if isinstance(model, G1TaskSpaceReachAtlas):
        reach = task_space_reach_from_target_torch(torch=torch, target_relative=target, model=model)
        output[:, 4:] = reach
    else:
        output[:, 4:11] = solve(
            inverse=model.left_damped_inverse,
            ready=model.left_ready_hand_relative_m,
            left=True,
        )
        output[:, 11:18] = solve(
            inverse=model.right_damped_inverse,
            ready=model.right_ready_hand_relative_m,
            left=False,
        )
    output[~(shot_active | predictive_ready)] = 0.0
    return output


def task_space_reach_from_target_torch(
    *,
    torch: Any,
    target_relative: Any,
    model: G1TaskSpaceReachAtlas,
) -> Any:
    """Interpolate bilateral normalized arm targets from a causal target."""

    if target_relative.ndim != 2 or target_relative.shape[1] != 3:
        raise ValueError("goalkeeper reach target must have shape (N, 3)")
    anchors = torch.as_tensor(
        model.target_relative_m,
        dtype=target_relative.dtype,
        device=target_relative.device,
    )
    scales = torch.as_tensor(
        model.interpolation_distance_scales_m,
        dtype=target_relative.dtype,
        device=target_relative.device,
    )
    canonical_target = target_relative.clone()
    canonical_target[:, 1] = torch.abs(canonical_target[:, 1])
    squared_distance = torch.sum(
        torch.square((canonical_target.unsqueeze(1) - anchors.unsqueeze(0)) / scales), dim=2
    )
    distance, neighbors = torch.topk(
        squared_distance, k=model.interpolation_neighbors, dim=1, largest=False
    )
    if model.interpolation_kernel == "gaussian":
        weights = torch.exp(-distance / model.interpolation_temperature)
    else:
        weights = 1.0 / (distance + 1.0e-4)
    weights /= torch.sum(weights, dim=1, keepdim=True)
    target_y = target_relative[:, 1]
    side_weight = torch.clamp(torch.abs(target_y) / model.arm_scale_transition_m, 0.0, 1.0)
    side_weight = side_weight * side_weight * (3.0 - 2.0 * side_weight)

    def one(actions: tuple[tuple[float, ...], ...], *, scale: float) -> Any:
        table = torch.as_tensor(actions, dtype=target_relative.dtype, device=target_relative.device)
        normalized = model.reach_gain * torch.sum(weights.unsqueeze(2) * table[neighbors], dim=1)
        arm_scale = model.central_support_scale + side_weight * (
            scale - model.central_support_scale
        )
        return torch.clamp(normalized * arm_scale.unsqueeze(1), -1.0, 1.0)

    canonical_left = one(model.left_normalized_action, scale=model.support_arm_scale)
    canonical_right = one(model.right_normalized_action, scale=model.primary_arm_scale)
    mirror_sign = torch.as_tensor(
        (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
        dtype=target_relative.dtype,
        device=target_relative.device,
    )
    reflected = torch.cat(
        (canonical_right * mirror_sign, canonical_left * mirror_sign), dim=1
    )
    canonical = torch.cat((canonical_left, canonical_right), dim=1)
    return torch.where((target_y < 0.0).unsqueeze(1), reflected, canonical)


def task_space_reach_from_target_numpy(
    *,
    target_relative: np.ndarray,
    model: G1TaskSpaceReachAtlas,
) -> np.ndarray:
    """NumPy parity path for independent CPU MuJoCo qualification."""

    target = np.asarray(target_relative, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 3 or not np.all(np.isfinite(target)):
        raise ValueError("goalkeeper reach target must have finite shape (N, 3)")
    anchors = np.asarray(model.target_relative_m, dtype=np.float64)
    scales = np.asarray(model.interpolation_distance_scales_m, dtype=np.float64)
    canonical_target = target.copy()
    canonical_target[:, 1] = np.abs(canonical_target[:, 1])
    squared_distance = np.sum(
        np.square((canonical_target[:, None, :] - anchors[None, :, :]) / scales), axis=2
    )
    neighbors = np.argsort(squared_distance, axis=1)[:, : model.interpolation_neighbors]
    distance = np.take_along_axis(squared_distance, neighbors, axis=1)
    if model.interpolation_kernel == "gaussian":
        weights = np.exp(-distance / model.interpolation_temperature)
    else:
        weights = 1.0 / (distance + 1.0e-4)
    weights /= np.sum(weights, axis=1, keepdims=True)
    target_y = target[:, 1]
    side_weight = np.clip(np.abs(target_y) / model.arm_scale_transition_m, 0.0, 1.0)
    side_weight = side_weight * side_weight * (3.0 - 2.0 * side_weight)

    def one(actions: tuple[tuple[float, ...], ...], *, scale: float) -> np.ndarray:
        table = np.asarray(actions, dtype=np.float64)
        normalized = model.reach_gain * np.sum(weights[:, :, None] * table[neighbors], axis=1)
        arm_scale = model.central_support_scale + side_weight * (
            scale - model.central_support_scale
        )
        return np.asarray(
            np.clip(normalized * arm_scale[:, None], -1.0, 1.0),
            dtype=np.float64,
        )

    canonical_left = one(model.left_normalized_action, scale=model.support_arm_scale)
    canonical_right = one(model.right_normalized_action, scale=model.primary_arm_scale)
    mirror_sign = np.asarray(
        (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0), dtype=np.float64
    )
    reflected = np.concatenate(
        (canonical_right * mirror_sign, canonical_left * mirror_sign), axis=1
    )
    canonical = np.concatenate((canonical_left, canonical_right), axis=1)
    return np.where((target_y < 0.0)[:, None], reflected, canonical)


def reach_model_payload(model: G1TaskSpaceReachModel | G1TaskSpaceReachAtlas) -> dict[str, Any]:
    payload = asdict(model)
    payload["model_hash"] = model.model_hash
    return payload


__all__ = [
    "G1TaskSpaceReachModel",
    "G1TaskSpaceReachAtlas",
    "GoalkeeperReachAtlasConfig",
    "GoalkeeperReachConfig",
    "build_g1_task_space_reach_atlas",
    "build_g1_task_space_reach_model",
    "reach_model_payload",
    "task_space_reach_from_target_numpy",
    "task_space_reach_from_target_torch",
    "task_space_reach_teacher_action",
]
