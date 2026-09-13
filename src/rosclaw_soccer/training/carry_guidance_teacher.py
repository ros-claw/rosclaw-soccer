"""Untrained, bounded G1 carry teacher for simulation data generation only.

This is not an autonomous learned skill or a controller installation. Callers
own foundation motion, actuator bounds, collision/body checks, evidence and
independent evaluation. Never use a teacher label as a success label.
"""

import math
from dataclasses import dataclass
from typing import Any

from rosclaw_soccer.training.ball_carry_credit import BallCarryCredit
from rosclaw_soccer.training.carry_task_features import carry_task_features


@dataclass(frozen=True)
class CarryGuidanceTeacherConfig:
    standoff_m: float = 0.20
    lane_heading_gain: float = 2.0
    lateral_foot_offset_m: float = 0.10

    def __post_init__(self) -> None:
        for value, lower, upper in (
            (self.standoff_m, 0.1, 0.35),
            (self.lane_heading_gain, 0.0, 4.0),
            (self.lateral_foot_offset_m, 0.0, 0.2),
        ):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not lower <= value <= upper
            ):
                raise ValueError("bounded finite simulation teacher geometry required")


_DEFAULT_CONFIG = CarryGuidanceTeacherConfig()


def carry_guidance_teacher(
    history: BallCarryCredit,
    *,
    ball_position: Any,
    root_position: Any,
    root_quaternion: Any,
    direction: Any,
    selected_foot: Any,
    config: CarryGuidanceTeacherConfig = _DEFAULT_CONFIG,
) -> Any:
    """Return world vx/vy/yaw-rate proposal, <=0.7 m/s and <=0.8 rad/s.

    Geometry is evaluated in float64; poses and the history share one measured
    float32/64 contract. A higher precision unit direction is accepted only if
    consistent with the history. No time-indexed motion or physical writes.
    """
    import torch

    if not isinstance(config, CarryGuidanceTeacherConfig):
        raise ValueError("explicit teacher config required")
    carry_task_features(
        history,
        ball_position=ball_position,
        root_position=root_position,
        root_quaternion=root_quaternion,
    )
    n = len(history.direction)
    if (
        not isinstance(direction, torch.Tensor)
        or direction.shape != (n, 2)
        or direction.dtype not in (torch.float32, torch.float64)
        or direction.device != history.direction.device
        or direction.layout != torch.strided
        or direction.requires_grad
        or not bool(torch.isfinite(direction).all())
        or not bool(
            torch.allclose(direction.double(), history.direction.double(), rtol=0, atol=1e-6)
        )
        or bool((torch.linalg.vector_norm(direction, dim=1) - 1).abs().gt(1e-6).any())
        or not isinstance(selected_foot, torch.Tensor)
        or selected_foot.shape != (n,)
        or selected_foot.dtype != torch.int64
        or selected_foot.device != direction.device
        or selected_foot.layout != torch.strided
        or not bool(((selected_foot == 0) | (selected_foot == 1)).all())
    ):
        raise ValueError("history-bound direction and binary selected-foot batch required")
    direction = direction.double()
    lateral = torch.stack((-direction[:, 1], direction[:, 0]), 1)
    # Preserve measured float32 subtraction before double geometry, as in the
    # qualified diagnostic; do not silently change the reference arithmetic.
    lane = ((ball_position[:, :2] - history.initial_ball[:, :2]) * lateral).sum(1)
    guide = direction - config.lane_heading_gain * lane.clamp(-0.5, 0.5)[:, None] * lateral
    guide = guide / torch.linalg.vector_norm(guide, dim=1)[:, None]
    guide_lateral = torch.stack((-guide[:, 1], guide[:, 0]), 1)
    waypoint = (
        ball_position[:, :2]
        - config.standoff_m * guide
        + config.lateral_foot_offset_m * (selected_foot.double() * 2 - 1)[:, None] * guide_lateral
    )
    xy = 2 * (waypoint - root_position[:, :2])
    magnitude = torch.linalg.vector_norm(xy, dim=1).clamp_min(1e-9)
    xy *= torch.minimum(torch.ones_like(magnitude), 0.7 / magnitude)[:, None]
    w, x, y, z = root_quaternion.double().unbind(1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    desired = torch.atan2(guide[:, 1], guide[:, 0])
    error = torch.atan2(torch.sin(desired - yaw), torch.cos(desired - yaw))
    return torch.cat((xy, error.clamp(-0.8, 0.8)[:, None]), 1)
