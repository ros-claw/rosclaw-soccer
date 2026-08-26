"""Backend-neutral agility shaping for a hierarchical goalkeeper.

The learned actor owns task intent while a frozen locomotion policy owns the
legs.  This module keeps those responsibilities explicit: it makes bounded
lateral intent more responsive, asks the locomotion prior to recenter between
shots, and attenuates only waist/arm residuals as root angular speed approaches
the naturalness ceiling.  It never emits torque or grants hardware authority.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class GoalkeeperAgilityConfig:
    """Immutable action-space contract shared by GPU training and CPU exam."""

    lateral_response_gain: float = 1.40
    recenter_deadband_m: float = 0.045
    recenter_scale_m: float = 0.42
    recenter_velocity_horizon_sec: float = 0.18
    angular_guard_onset_rad_s: float = 1.50
    angular_guard_ceiling_rad_s: float = 2.80
    minimum_upper_body_scale: float = 0.16
    waist_authority_scale: float = 0.55
    arm_authority_scale: float = 0.70
    counter_rotation_gain: float = 0.16
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.goalkeeper_agility_config.v1"

    def __post_init__(self) -> None:
        positive = (
            self.lateral_response_gain,
            self.recenter_deadband_m,
            self.recenter_scale_m,
            self.recenter_velocity_horizon_sec,
            self.angular_guard_onset_rad_s,
            self.angular_guard_ceiling_rad_s,
            self.minimum_upper_body_scale,
            self.waist_authority_scale,
            self.arm_authority_scale,
            self.counter_rotation_gain,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("goalkeeper agility settings must be finite and positive")
        if not 1.0 <= self.lateral_response_gain <= 3.0:
            raise ValueError("goalkeeper lateral response gain is outside [1, 3]")
        if not self.recenter_deadband_m < self.recenter_scale_m <= 1.0:
            raise ValueError("goalkeeper recenter geometry is invalid")
        if not 0.05 <= self.recenter_velocity_horizon_sec <= 0.50:
            raise ValueError("goalkeeper recenter velocity horizon is invalid")
        if not self.angular_guard_onset_rad_s < self.angular_guard_ceiling_rad_s:
            raise ValueError("goalkeeper angular guard interval is invalid")
        if not 0.10 <= self.minimum_upper_body_scale <= 0.80:
            raise ValueError("goalkeeper upper-body scale floor is invalid")
        if not 0.25 <= self.waist_authority_scale <= 0.80:
            raise ValueError("goalkeeper waist authority scale is invalid")
        if not 0.40 <= self.arm_authority_scale <= 0.90:
            raise ValueError("goalkeeper arm authority scale is invalid")
        if not 0.0 < self.counter_rotation_gain <= 0.25:
            raise ValueError("goalkeeper counter-rotation gain is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("goalkeeper agility shaping is SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def shape_goalkeeper_action_numpy(
    *,
    requested_action: NDArray[np.float64],
    root_lateral_position_m: NDArray[np.float64],
    root_lateral_velocity_mps: NDArray[np.float64],
    root_angular_velocity_rad_s: NDArray[np.float64],
    shot_active: NDArray[np.bool_],
    config: GoalkeeperAgilityConfig | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return desired actor action and upper-body authority scale."""

    cfg = config or GoalkeeperAgilityConfig()
    action = np.asarray(requested_action, dtype=np.float64)
    position = np.asarray(root_lateral_position_m, dtype=np.float64)
    velocity = np.asarray(root_lateral_velocity_mps, dtype=np.float64)
    angular = np.asarray(root_angular_velocity_rad_s, dtype=np.float64)
    active = np.asarray(shot_active, dtype=np.bool_)
    count = action.shape[0] if action.ndim == 2 else -1
    if (
        action.ndim != 2
        or action.shape[1] != 18
        or position.shape != (count,)
        or velocity.shape != (count,)
        or angular.shape != (count, 3)
        or active.shape != (count,)
    ):
        raise ValueError("goalkeeper agility NumPy shapes are invalid")
    if not all(np.all(np.isfinite(value)) for value in (action, position, velocity, angular)):
        raise ValueError("goalkeeper agility NumPy inputs must be finite")
    desired = np.clip(action.copy(), -1.0, 1.0)
    denominator = math.tanh(cfg.lateral_response_gain)
    desired[active, 0] = np.tanh(cfg.lateral_response_gain * desired[active, 0]) / denominator
    predicted_lateral = position + cfg.recenter_velocity_horizon_sec * velocity
    recenter = np.clip(predicted_lateral / cfg.recenter_scale_m, -1.0, 1.0)
    recenter[np.abs(predicted_lateral) <= cfg.recenter_deadband_m] = 0.0
    desired[~active] = 0.0
    desired[~active, 0] = recenter[~active]
    angular_speed = np.linalg.norm(angular, axis=1)
    fraction = np.clip(
        (angular_speed - cfg.angular_guard_onset_rad_s)
        / (cfg.angular_guard_ceiling_rad_s - cfg.angular_guard_onset_rad_s),
        0.0,
        1.0,
    )
    upper_scale = 1.0 - fraction * (1.0 - cfg.minimum_upper_body_scale)
    desired[:, 1:] *= upper_scale[:, None]
    desired[:, 1:4] *= cfg.waist_authority_scale
    desired[:, 4:] *= cfg.arm_authority_scale
    # Waist residual order is yaw, roll, pitch while root angular velocity is
    # x, y, z.  Apply a small velocity-opposing reflex only during a shot and
    # only as the guard activates; the inactive phase remains fully owned by
    # the frozen locomotion prior.
    counter = np.stack((-angular[:, 2], -angular[:, 0], -angular[:, 1]), axis=1)
    desired[:, 1:4] += counter * (cfg.counter_rotation_gain * fraction * active)[:, None]
    desired = np.clip(desired, -1.0, 1.0)
    return desired, upper_scale


def shape_goalkeeper_action_torch(
    *,
    requested_action: Any,
    root_lateral_position_m: Any,
    root_lateral_velocity_mps: Any,
    root_angular_velocity_rad_s: Any,
    shot_active: Any,
    config: GoalkeeperAgilityConfig | None = None,
) -> tuple[Any, Any]:
    """Torch equivalent of :func:`shape_goalkeeper_action_numpy`."""

    import torch

    cfg = config or GoalkeeperAgilityConfig()
    count = requested_action.shape[0] if requested_action.ndim == 2 else -1
    if (
        tuple(requested_action.shape) != (count, 18)
        or tuple(root_lateral_position_m.shape) != (count,)
        or tuple(root_lateral_velocity_mps.shape) != (count,)
        or tuple(root_angular_velocity_rad_s.shape) != (count, 3)
        or tuple(shot_active.shape) != (count,)
    ):
        raise ValueError("goalkeeper agility Torch shapes are invalid")
    if not bool(
        torch.all(torch.isfinite(requested_action))
        and torch.all(torch.isfinite(root_lateral_position_m))
        and torch.all(torch.isfinite(root_lateral_velocity_mps))
        and torch.all(torch.isfinite(root_angular_velocity_rad_s))
    ):
        raise ValueError("goalkeeper agility Torch inputs must be finite")
    active = shot_active.to(torch.bool)
    desired = torch.clamp(requested_action, -1.0, 1.0).clone()
    shaped_lateral = torch.tanh(cfg.lateral_response_gain * desired[:, 0]) / math.tanh(
        cfg.lateral_response_gain
    )
    predicted_lateral = (
        root_lateral_position_m + cfg.recenter_velocity_horizon_sec * root_lateral_velocity_mps
    )
    recenter = torch.clamp(predicted_lateral / cfg.recenter_scale_m, -1.0, 1.0)
    recenter = torch.where(
        torch.abs(predicted_lateral) <= cfg.recenter_deadband_m,
        torch.zeros_like(recenter),
        recenter,
    )
    desired = torch.where(active.unsqueeze(1), desired, torch.zeros_like(desired))
    desired[:, 0] = torch.where(active, shaped_lateral, recenter)
    angular_speed = torch.linalg.vector_norm(root_angular_velocity_rad_s, dim=1)
    fraction = torch.clamp(
        (angular_speed - cfg.angular_guard_onset_rad_s)
        / (cfg.angular_guard_ceiling_rad_s - cfg.angular_guard_onset_rad_s),
        0.0,
        1.0,
    )
    upper_scale = 1.0 - fraction * (1.0 - cfg.minimum_upper_body_scale)
    desired[:, 1:] *= upper_scale.unsqueeze(1)
    desired[:, 1:4] *= cfg.waist_authority_scale
    desired[:, 4:] *= cfg.arm_authority_scale
    counter = torch.stack(
        (
            -root_angular_velocity_rad_s[:, 2],
            -root_angular_velocity_rad_s[:, 0],
            -root_angular_velocity_rad_s[:, 1],
        ),
        dim=1,
    )
    desired[:, 1:4] += counter * (
        cfg.counter_rotation_gain * fraction * active.to(requested_action.dtype)
    ).unsqueeze(1)
    desired = torch.clamp(desired, -1.0, 1.0)
    return desired, upper_scale


__all__ = [
    "GoalkeeperAgilityConfig",
    "shape_goalkeeper_action_numpy",
    "shape_goalkeeper_action_torch",
]
