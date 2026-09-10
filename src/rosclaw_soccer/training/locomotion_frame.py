"""Explicit idle-phase frame selection for mirrored locomotion proposals.

Tiny lateral commands can otherwise alternate a recurrent model's coordinate
frame. This is NOT universal hysteresis: a caller must explicitly identify a
phase compatible with the model's declared idle frame. Outside that phase the
legacy sign rule is unchanged. No command, joint target, recurrent state,
simulation, admission, or execution authority is owned here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LocomotionFrameConfig:
    idle_deadband_mps: float = 0.02
    idle_reflected: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.idle_deadband_mps) not in (float, int)
            or not math.isfinite(self.idle_deadband_mps)
            or not 0 <= self.idle_deadband_mps <= 0.1
            or type(self.idle_reflected) is not bool
        ):
            raise ValueError("bounded idle-frame configuration required")


def _config(config: LocomotionFrameConfig | None) -> LocomotionFrameConfig:
    if config is not None and not isinstance(config, LocomotionFrameConfig):
        raise ValueError("explicit locomotion frame configuration required")
    return config or LocomotionFrameConfig()


def select_locomotion_reflection(
    lateral_velocity_mps: float,
    *,
    prefer_idle_frame: bool,
    config: LocomotionFrameConfig | None = None,
) -> bool:
    """Select a frame, not a motion command; caller owns phase applicability.

    Keeping a previously mirrored frame near zero is deliberately NOT the rule:
    the learned motor may depend on the original unreflected idle convention.
    """
    active = _config(config)
    if (
        type(lateral_velocity_mps) not in (float, int)
        or not math.isfinite(lateral_velocity_mps)
        or abs(lateral_velocity_mps) > 10
        or type(prefer_idle_frame) is not bool
    ):
        raise ValueError("finite lateral velocity and explicit idle-phase flag required")
    if prefer_idle_frame and abs(lateral_velocity_mps) <= active.idle_deadband_mps:
        return active.idle_reflected
    return lateral_velocity_mps < -1e-6


def select_locomotion_reflection_batch(
    lateral_velocity_mps: Any,
    *,
    prefer_idle_frame: Any,
    config: LocomotionFrameConfig | None = None,
) -> Any:
    """Detached float32/64 velocities and same-device boolean phase flags.

    Returns a new boolean tensor without changing commands or model histories.
    Torch is optional until this batched training helper is invoked.
    """
    import torch

    active = _config(config)
    velocity, idle = lateral_velocity_mps, prefer_idle_frame
    if (
        not isinstance(velocity, torch.Tensor)
        or velocity.ndim != 1
        or not 1 <= len(velocity) <= 4096
        or velocity.dtype not in (torch.float32, torch.float64)
        or velocity.layout != torch.strided
        or velocity.requires_grad
        or not bool(torch.isfinite(velocity).all())
        or bool((velocity.abs() > 10).any())
        or not isinstance(idle, torch.Tensor)
        or idle.shape != velocity.shape
        or idle.dtype != torch.bool
        or idle.layout != torch.strided
        or idle.device != velocity.device
    ):
        raise ValueError("finite detached velocity and aligned boolean idle-phase tensors required")
    return torch.where(
        idle & (velocity.abs() <= active.idle_deadband_mps),
        active.idle_reflected,
        velocity < -1e-6,
    )
