"""Immutable, player-bound SIM_ONLY foot geometry for measured motor policies."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np


def _plain_finite_tuple(value: Any, shape: tuple[int, ...]) -> bool:
    if not shape:
        return type(value) in (int, float) and math.isfinite(value)
    return (
        type(value) is tuple
        and len(value) == shape[0]
        and all(_plain_finite_tuple(item, shape[1:]) for item in value)
    )


@dataclass(frozen=True)
class TeamFootKinematics:
    agent_id: str
    frame: int
    foot_position_world_m: tuple[tuple[float, ...], ...]
    foot_linear_jacobian_world: tuple[tuple[tuple[float, ...], ...], ...]
    leg_joint_limits_rad: tuple[tuple[tuple[float, ...], ...], ...]

    def __post_init__(self) -> None:
        if not (
            _plain_finite_tuple(self.foot_position_world_m, (2, 3))
            and _plain_finite_tuple(self.foot_linear_jacobian_world, (2, 3, 6))
            and _plain_finite_tuple(self.leg_joint_limits_rad, (2, 6, 2))
        ):
            raise ValueError("immutable finite foot kinematics required")
        feet = np.asarray(self.foot_position_world_m, dtype=float)
        jacobian = np.asarray(self.foot_linear_jacobian_world, dtype=float)
        limits = np.asarray(self.leg_joint_limits_rad, dtype=float)
        if (
            not isinstance(self.agent_id, str)
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", self.agent_id) is None
            or type(self.frame) is not int
            or self.frame < 0
            or feet.shape != (2, 3)
            or jacobian.shape != (2, 3, 6)
            or limits.shape != (2, 6, 2)
            or np.any(limits[:, :, 0] >= limits[:, :, 1])
            or np.max(np.abs(feet)) > 1000
            or np.max(np.abs(jacobian)) > 1000
        ):
            raise ValueError("finite player-bound foot kinematics required")


def measure_team_foot_kinematics(
    *,
    model: Any,
    data: Any,
    agent_id: str,
    frame: int,
    ankle_body_ids: tuple[int, int],
    leg_dof_ids: tuple[tuple[int, ...], tuple[int, ...]],
    leg_joint_ranges: tuple[tuple[tuple[float, ...], ...], ...],
) -> TeamFootKinematics:
    """Copy MuJoCo geometry; never expose model/data or a writable actuator handle."""
    import mujoco

    if (
        len(ankle_body_ids) != 2
        or any(len(ids) != 6 for ids in leg_dof_ids)
        or any(body < 0 or body >= model.nbody for body in ankle_body_ids)
        or any(dof < 0 or dof >= model.nv for ids in leg_dof_ids for dof in ids)
    ):
        raise ValueError("invalid G1 foot body or six-leg DoF mapping")
    jacobians = []
    for body_id, dofs in zip(ankle_body_ids, leg_dof_ids, strict=True):
        linear = np.zeros((3, model.nv))
        angular = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, linear, angular, body_id)
        jacobians.append(tuple(tuple(float(value) for value in row[list(dofs)]) for row in linear))
    measured = TeamFootKinematics(
        agent_id=agent_id,
        frame=frame,
        foot_position_world_m=tuple(
            tuple(float(value) for value in data.xpos[body_id]) for body_id in ankle_body_ids
        ),
        foot_linear_jacobian_world=tuple(jacobians),
        leg_joint_limits_rad=leg_joint_ranges,
    )
    return measured
