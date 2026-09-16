"""Bounded planar target-tracking proposals; no simulator or motion authority.

All coordinates and velocities share one declared world frame. The caller owns
effector selection, observation timing, frame conversion, native body limits,
history and execution guards. This numeric primitive never declares a catch.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class PlanarInterceptionConfig:
    desired_offset_xy_m: tuple[float, float]
    position_gain_per_sec: float
    maximum_speed_mps: float
    maximum_delta_mps_per_call: float
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        if (
            type(self.desired_offset_xy_m) is not tuple
            or len(self.desired_offset_xy_m) != 2
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 10
                for v in self.desired_offset_xy_m
            )
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (
                    self.position_gain_per_sec,
                    self.maximum_speed_mps,
                    self.maximum_delta_mps_per_call,
                )
            )
            or not 0 < self.position_gain_per_sec <= 20
            or not 0 < self.maximum_speed_mps <= 5
            or not 0 < self.maximum_delta_mps_per_call <= self.maximum_speed_mps
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("explicit bounded SIM_ONLY interception configuration required")

    @property
    def config_hash(self) -> str:
        return str(hash_json({"schema": "soccer.planar_interception.v1", **asdict(self)}))


def propose_planar_interception(
    *,
    target_position_xy: Any,
    target_velocity_xy: Any,
    effector_position_xy: Any,
    previous_velocity_xy: Any,
    config: PlanarInterceptionConfig,
) -> Any:
    """Return a fresh float32 world-velocity tensor with speed/delta bounds.

    Torch is optional until called. Inputs must be detached, aligned snapshots;
    no observation or caller history is mutated. The explicit per-call delta
    is not an acceleration guarantee without a verified caller clock. Bounds
    apply to this proposal, not measured robot velocity or collision safety.
    """
    import torch

    if not isinstance(config, PlanarInterceptionConfig):
        raise ValueError("typed interception configuration required")
    config.__post_init__()
    values = (target_position_xy, target_velocity_xy, effector_position_xy, previous_velocity_xy)
    if any(
        not isinstance(v, torch.Tensor)
        or v.shape != (2,)
        or v.dtype != torch.float32
        or v.layout != torch.strided
        or v.requires_grad
        or not bool(torch.isfinite(v).all())
        or bool((v.abs() > 1000).any())
        for v in values
    ) or any(v.device != target_position_xy.device for v in values):
        raise ValueError("finite detached same-device float32 planar snapshots required")
    if float(torch.linalg.vector_norm(previous_velocity_xy)) > config.maximum_speed_mps + 1e-6:
        raise ValueError("previous proposal exceeds the declared speed envelope")
    desired_position = target_position_xy + target_position_xy.new_tensor(
        config.desired_offset_xy_m
    )
    desired = target_velocity_xy + config.position_gain_per_sec * (
        desired_position - effector_position_xy
    )
    desired = desired * (
        config.maximum_speed_mps / torch.linalg.vector_norm(desired).clamp_min(1e-9)
    ).clamp_max(1)
    delta = desired - previous_velocity_xy
    delta = delta * (
        config.maximum_delta_mps_per_call / torch.linalg.vector_norm(delta).clamp_min(1e-9)
    ).clamp_max(1)
    result = previous_velocity_xy + delta
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("nonfinite interception proposal")
    return result
