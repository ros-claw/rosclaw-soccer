"""Pose-coherent waist-and-arm reach atlas for hard low goalkeeper shots.

The legacy reach atlas solves the two arms around an upright ready pose and
then combines those labels with a separately manufactured crouch.  That is a
poor teacher for low far-corner shots: the hand, waist and pelvis targets do
not describe one kinematically consistent body.  This module solves the waist
roll/pitch and the target-side seven-joint arm together on the exact qualified
G1 MuJoCo asset.  It remains a bounded, content-bound ``SIM_ONLY`` teacher.
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

from rosclaw_soccer.providers.g1.asset_qualification import g1_body_hash
from rosclaw_soccer.sim.contracts import hash_json

_SUBSPACE_MIRROR_ORDER = (0, 1, 2, 10, 11, 12, 13, 14, 15, 16, 3, 4, 5, 6, 7, 8, 9)
_SUBSPACE_MIRROR_SIGN = (
    -1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
)
_ARM_SOFT_LIMITS_RAD = np.asarray((0.70, 0.75, 0.55, 0.55, 0.18, 0.16, 0.16), dtype=np.float64)
_ARM_MIRROR_SIGN = np.asarray((1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0))


@dataclass(frozen=True)
class GoalkeeperWholeBodyReachConfig:
    """Deterministic nonlinear IK and interpolation contract."""

    iterations: int = 128
    multistart_count: int = 12
    multistart_seed: int = 45_101
    damping: float = 0.06
    step_limit_rad: float = 0.08
    error_tolerance_m: float = 0.008
    target_x_m: float = -0.08
    lateral_targets_m: tuple[float, ...] = (0.0, 0.30, 0.45, 0.60, 0.75)
    relative_height_targets_m: tuple[float, ...] = (-0.40, -0.25, -0.10, 0.10, 0.35, 0.60)
    waist_roll_limit_rad: float = 0.45
    waist_pitch_limit_rad: float = 0.35
    arm_workspace_scale: float = 2.50
    support_counterbalance_scale: float = 0.65
    interpolation_distance_scales_m: tuple[float, float, float] = (0.24, 0.20, 0.18)
    interpolation_neighbors: int = 8
    interpolation_temperature: float = 0.65
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_whole_body_reach_config.v2"

    def __post_init__(self) -> None:
        if not 32 <= self.iterations <= 256 or not 1 <= self.multistart_count <= 32:
            raise ValueError("whole-body reach solver budget is invalid")
        if not 0 <= self.multistart_seed < 2**31:
            raise ValueError("whole-body reach seed is invalid")
        if not 0.03 <= self.damping <= 0.20 or not 0.02 <= self.step_limit_rad <= 0.15:
            raise ValueError("whole-body reach solver settings are invalid")
        if not 0.002 <= self.error_tolerance_m <= 0.03:
            raise ValueError("whole-body reach tolerance is invalid")
        if not -0.30 <= self.target_x_m <= 0.10:
            raise ValueError("whole-body reach forward target is invalid")
        lateral = np.asarray(self.lateral_targets_m, dtype=np.float64)
        height = np.asarray(self.relative_height_targets_m, dtype=np.float64)
        if (
            lateral.ndim != 1
            or height.ndim != 1
            or lateral.size < 4
            or height.size < 4
            or lateral[0] != 0.0
            or np.any(np.diff(lateral) <= 0.0)
            or np.any(np.diff(height) <= 0.0)
            or not np.all(np.isfinite(lateral))
            or not np.all(np.isfinite(height))
        ):
            raise ValueError("whole-body reach grid is invalid")
        values = (
            self.waist_roll_limit_rad,
            self.waist_pitch_limit_rad,
            self.arm_workspace_scale,
            self.interpolation_temperature,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("whole-body reach settings must be finite and positive")
        if not 0.20 <= self.waist_roll_limit_rad <= 0.50:
            raise ValueError("whole-body reach waist-roll limit is invalid")
        if not 0.15 <= self.waist_pitch_limit_rad <= 0.40:
            raise ValueError("whole-body reach waist-pitch limit is invalid")
        if not 1.0 <= self.arm_workspace_scale <= 2.5:
            raise ValueError("whole-body reach arm workspace is invalid")
        if not math.isfinite(self.support_counterbalance_scale) or not (
            0.0 <= self.support_counterbalance_scale <= 1.0
        ):
            raise ValueError("whole-body reach support counterbalance is invalid")
        scales = np.asarray(self.interpolation_distance_scales_m, dtype=np.float64)
        anchor_count = lateral.size * height.size
        if scales.shape != (3,) or np.any(scales <= 0.0) or not np.all(np.isfinite(scales)):
            raise ValueError("whole-body reach interpolation scales are invalid")
        if not 1 <= self.interpolation_neighbors <= anchor_count:
            raise ValueError("whole-body reach neighbor count is invalid")
        if not 0.10 <= self.interpolation_temperature <= 4.0:
            raise ValueError("whole-body reach temperature is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("whole-body reach teacher must remain SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class G1WholeBodyReachAtlas:
    """Content-bound waist/arm joint deltas for canonical positive targets."""

    body_hash: str
    config_hash: str
    target_relative_m: tuple[tuple[float, float, float], ...]
    joint_delta_rad: tuple[tuple[float, ...], ...]
    terminal_error_m: tuple[float, ...]
    interpolation_distance_scales_m: tuple[float, float, float]
    interpolation_neighbors: int
    interpolation_temperature: float
    joint_authority: str = "WAIST_ROLL_PITCH_PLUS_TARGET_SIDE_ARM_TEACHER_ONLY"
    interpolation_symmetry: str = "POSITIVE_HALF_SPACE_EXACT_REFLECTION_V1"
    physics_authority: bool = False
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.g1_whole_body_reach_atlas.v1"

    def __post_init__(self) -> None:
        targets = np.asarray(self.target_relative_m, dtype=np.float64)
        delta = np.asarray(self.joint_delta_rad, dtype=np.float64)
        errors = np.asarray(self.terminal_error_m, dtype=np.float64)
        if not self.body_hash.startswith("sha256:") or not self.config_hash.startswith("sha256:"):
            raise ValueError("whole-body reach atlas requires content hashes")
        if (
            targets.ndim != 2
            or targets.shape[1] != 3
            or targets.shape[0] < 16
            or delta.shape != (targets.shape[0], 17)
            or errors.shape != (targets.shape[0],)
            or np.any(targets[:, 1] < 0.0)
            or not np.all(np.isfinite(targets))
            or not np.all(np.isfinite(delta))
            or not np.all(np.isfinite(errors))
            or np.any(errors < 0.0)
        ):
            raise ValueError("whole-body reach atlas tensors are invalid")
        if not 1 <= self.interpolation_neighbors <= targets.shape[0]:
            raise ValueError("whole-body reach atlas neighbor count is invalid")
        if self.physics_authority or self.activation_ceiling != "SIM_ONLY":
            raise ValueError("whole-body reach atlas exceeds its authority")

    @property
    def model_hash(self) -> str:
        return str(hash_json(asdict(self)))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "model_hash": self.model_hash}


def load_g1_whole_body_reach_atlas(path: Path) -> G1WholeBodyReachAtlas:
    """Load one numeric-only, tamper-evident reach teacher."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file() or not 1 <= resolved.stat().st_size <= 2 * 1024 * 1024:
        raise ValueError("whole-body reach atlas is missing, empty, or too large")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("whole-body reach atlas root must be an object")
    claimed_hash = payload.pop("model_hash", None)
    try:
        payload["target_relative_m"] = tuple(
            tuple(float(value) for value in row) for row in payload["target_relative_m"]
        )
        payload["joint_delta_rad"] = tuple(
            tuple(float(value) for value in row) for row in payload["joint_delta_rad"]
        )
        payload["terminal_error_m"] = tuple(float(value) for value in payload["terminal_error_m"])
        payload["interpolation_distance_scales_m"] = tuple(
            float(value) for value in payload["interpolation_distance_scales_m"]
        )
        atlas = G1WholeBodyReachAtlas(**payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("whole-body reach atlas payload is invalid") from exc
    if claimed_hash != atlas.model_hash:
        raise ValueError("whole-body reach atlas content hash mismatch")
    return atlas


def write_g1_whole_body_reach_atlas(
    atlas: G1WholeBodyReachAtlas,
    path: Path,
) -> Path:
    """Persist an atlas atomically as JSON numeric data, never pickle."""

    output = path.expanduser().resolve()
    if output.exists():
        raise ValueError("whole-body reach atlas output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(atlas.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return output


def build_g1_whole_body_reach_atlas(
    asset_root: Path,
    *,
    config: GoalkeeperWholeBodyReachConfig | None = None,
) -> G1WholeBodyReachAtlas:
    """Solve a deterministic joint-bounded low-reach atlas on the exact G1."""

    import mujoco

    from rosclaw_soccer.training.goalkeeper_mjwarp import _LOCO_DEFAULT, _LOCO_TO_MOTOR
    from rosclaw_soccer.world.field import build_g1_stadium_model

    active = config or GoalkeeperWholeBodyReachConfig()
    root = asset_root.expanduser().resolve()
    model = build_g1_stadium_model(root)
    data = mujoco.MjData(model)
    ready = np.zeros(29, dtype=np.float64)
    ready[np.asarray(_LOCO_TO_MOTOR, dtype=np.int64)] = np.asarray(_LOCO_DEFAULT, dtype=np.float64)
    arm_limits = _ARM_SOFT_LIMITS_RAD * 0.88 * active.arm_workspace_scale
    subspace_motors = np.asarray(tuple(range(12, 29)), dtype=np.int64)
    active_motors = np.asarray((13, 14, 22, 23, 24, 25, 26, 27, 28), dtype=np.int64)
    qpos_indices = 7 + active_motors
    dof_indices = 6 + active_motors
    soft_limits = np.concatenate(
        ((active.waist_roll_limit_rad, active.waist_pitch_limit_rad), arm_limits)
    )
    lower = ready[active_motors] - soft_limits
    upper = ready[active_motors] + soft_limits
    for offset, qpos_index in enumerate(qpos_indices):
        joint_candidates = np.flatnonzero(model.jnt_qposadr == qpos_index)
        if joint_candidates.size != 1:
            raise RuntimeError("qualified G1 whole-body joint mapping changed")
        joint = int(joint_candidates[0])
        if bool(model.jnt_limited[joint]):
            lower[offset] = max(lower[offset], float(model.jnt_range[joint, 0]))
            upper[offset] = min(upper[offset], float(model.jnt_range[joint, 1]))
    targets = tuple(
        (active.target_x_m, float(y), float(z))
        for y in active.lateral_targets_m
        for z in active.relative_height_targets_m
    )
    geom = int(model.geom("right_hand_collision").id)
    solutions: list[tuple[float, ...]] = []
    terminal_errors: list[float] = []
    for target_index, target in enumerate(targets):
        target_array = np.asarray(target, dtype=np.float64)
        rng = np.random.default_rng(active.multistart_seed + 7_919 * target_index)
        best_error = math.inf
        best_position = ready[active_motors].copy()
        for restart in range(active.multistart_count):
            data.qpos.fill(0.0)
            data.qvel.fill(0.0)
            data.qpos[:7] = (4.52, 0.0, 0.793, 0.0, 0.0, 0.0, 1.0)
            data.qpos[7:36] = ready
            if restart > 0:
                data.qpos[qpos_indices] = rng.uniform(lower, upper)
            for _ in range(active.iterations):
                mujoco.mj_forward(model, data)
                relative = np.asarray(data.geom_xpos[geom] - data.qpos[:3], dtype=np.float64)
                error = target_array - relative
                if float(np.linalg.norm(error)) <= active.error_tolerance_m:
                    break
                jacobian = np.zeros((3, model.nv), dtype=np.float64)
                angular = np.zeros_like(jacobian)
                mujoco.mj_jacGeom(model, data, jacobian, angular, geom)
                jacobian = jacobian[:, dof_indices]
                delta = (
                    jacobian.T
                    @ np.linalg.inv(jacobian @ jacobian.T + active.damping**2 * np.eye(3))
                    @ error
                )
                delta[:2] *= 0.45
                data.qpos[qpos_indices] = np.clip(
                    data.qpos[qpos_indices]
                    + np.clip(delta, -active.step_limit_rad, active.step_limit_rad),
                    lower,
                    upper,
                )
            mujoco.mj_forward(model, data)
            relative = np.asarray(data.geom_xpos[geom] - data.qpos[:3], dtype=np.float64)
            terminal_error = float(np.linalg.norm(target_array - relative))
            if terminal_error < best_error:
                best_error = terminal_error
                best_position = data.qpos[qpos_indices].copy()
        full_delta = np.zeros(17, dtype=np.float64)
        active_subspace = active_motors - subspace_motors[0]
        full_delta[active_subspace] = best_position - ready[active_motors]
        # The target-side arm owns the interception.  A mirrored, bounded
        # support-arm swing counters its roll impulse on the floating base;
        # without it, static IK improves reach while dynamic physics topples
        # the goalkeeper.  The support arm is teacher data, not a motor-side
        # heuristic, and remains part of the content hash.
        full_delta[3:10] = (
            active.support_counterbalance_scale * full_delta[10:17] * _ARM_MIRROR_SIGN
        )
        solutions.append(tuple(float(value) for value in full_delta))
        terminal_errors.append(best_error)
    return G1WholeBodyReachAtlas(
        body_hash=g1_body_hash(root),
        config_hash=active.config_hash,
        target_relative_m=targets,
        joint_delta_rad=tuple(solutions),
        terminal_error_m=tuple(terminal_errors),
        interpolation_distance_scales_m=active.interpolation_distance_scales_m,
        interpolation_neighbors=active.interpolation_neighbors,
        interpolation_temperature=active.interpolation_temperature,
    )


def whole_body_reach_from_target_torch(
    *, torch: Any, target_relative: Any, model: G1WholeBodyReachAtlas
) -> Any:
    """Interpolate waist/arm deltas with exact left/right reflection."""

    if target_relative.ndim != 2 or target_relative.shape[1] != 3:
        raise ValueError("whole-body reach target must have shape (N, 3)")
    anchors = torch.as_tensor(
        model.target_relative_m, dtype=target_relative.dtype, device=target_relative.device
    )
    scales = torch.as_tensor(
        model.interpolation_distance_scales_m,
        dtype=target_relative.dtype,
        device=target_relative.device,
    )
    canonical = target_relative.clone()
    canonical[:, 1] = torch.abs(canonical[:, 1])
    squared_distance = torch.sum(
        torch.square((canonical.unsqueeze(1) - anchors.unsqueeze(0)) / scales), dim=2
    )
    distance, neighbors = torch.topk(
        squared_distance, k=model.interpolation_neighbors, dim=1, largest=False
    )
    weights = torch.exp(-distance / model.interpolation_temperature)
    weights /= torch.sum(weights, dim=1, keepdim=True)
    table = torch.as_tensor(
        model.joint_delta_rad, dtype=target_relative.dtype, device=target_relative.device
    )
    positive = torch.sum(weights.unsqueeze(2) * table[neighbors], dim=1)
    order = torch.as_tensor(_SUBSPACE_MIRROR_ORDER, dtype=torch.long, device=target_relative.device)
    sign = torch.as_tensor(
        _SUBSPACE_MIRROR_SIGN, dtype=target_relative.dtype, device=target_relative.device
    )
    reflected = positive[:, order] * sign
    return torch.where((target_relative[:, 1] < 0.0).unsqueeze(1), reflected, positive)


def whole_body_reach_from_target_numpy(
    *, target_relative: NDArray[np.float64], model: G1WholeBodyReachAtlas
) -> NDArray[np.float64]:
    """NumPy twin of the training interpolator for SIM-only deployment exams."""

    target = np.asarray(target_relative, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 3 or not np.isfinite(target).all():
        raise ValueError("whole-body reach target must have finite shape (N, 3)")
    anchors = np.asarray(model.target_relative_m, dtype=np.float64)
    scales = np.asarray(model.interpolation_distance_scales_m, dtype=np.float64)
    canonical = target.copy()
    canonical[:, 1] = np.abs(canonical[:, 1])
    squared_distance = np.sum(
        np.square((canonical[:, None, :] - anchors[None, :, :]) / scales),
        axis=2,
    )
    neighbors = np.argpartition(
        squared_distance,
        kth=model.interpolation_neighbors - 1,
        axis=1,
    )[:, : model.interpolation_neighbors]
    distance = np.take_along_axis(squared_distance, neighbors, axis=1)
    weights = np.exp(-distance / model.interpolation_temperature)
    weights /= np.sum(weights, axis=1, keepdims=True)
    table = np.asarray(model.joint_delta_rad, dtype=np.float64)
    positive = np.sum(weights[:, :, None] * table[neighbors], axis=1)
    reflected = positive[:, np.asarray(_SUBSPACE_MIRROR_ORDER)] * np.asarray(
        _SUBSPACE_MIRROR_SIGN,
        dtype=np.float64,
    )
    return np.where((target[:, 1] < 0.0)[:, None], reflected, positive)


__all__ = [
    "G1WholeBodyReachAtlas",
    "GoalkeeperWholeBodyReachConfig",
    "build_g1_whole_body_reach_atlas",
    "load_g1_whole_body_reach_atlas",
    "whole_body_reach_from_target_numpy",
    "whole_body_reach_from_target_torch",
    "write_g1_whole_body_reach_atlas",
]
