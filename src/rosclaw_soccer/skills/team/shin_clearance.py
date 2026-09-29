"""Read-only, same-player shin-to-ball distance differential in MuJoCo simulation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TeamShinClearance:
    """Measured signed gaps and local leg-joint gradients; no simulator handle."""

    agent_id: str
    frame: int
    clearance_m: tuple[float, float]
    gradient_m_per_rad: tuple[tuple[float, ...], tuple[float, ...]]

    def __post_init__(self) -> None:
        if (
            type(self.agent_id) is not str
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", self.agent_id) is None
            or type(self.frame) is not int
            or not 0 <= self.frame < 1000
            or type(self.clearance_m) is not tuple
            or len(self.clearance_m) != 2
            or type(self.gradient_m_per_rad) is not tuple
            or len(self.gradient_m_per_rad) != 2
            or any(type(row) is not tuple or len(row) != 6 for row in self.gradient_m_per_rad)
            or any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in (
                    *self.clearance_m,
                    *self.gradient_m_per_rad[0],
                    *self.gradient_m_per_rad[1],
                )
            )
            or any(not -1 <= value <= 100 for value in self.clearance_m)
            or any(abs(value) > 100 for row in self.gradient_m_per_rad for value in row)
        ):
            raise ValueError("finite same-player measured shin clearance required")


def measure_team_shin_clearance(
    *,
    model: Any,
    data: Any,
    agent_id: str,
    frame: int,
    ball_geom_id: int,
    leg_dof_ids: tuple[tuple[int, ...], tuple[int, ...]],
) -> TeamShinClearance:
    """Copy MuJoCo signed gaps and a local analytic Jacobian before control.

    The gradient is d(shin-ball signed distance)/d(leg joint), holding the
    current ball pose fixed. It is a local differential, not future contact
    evidence; all normal execution and collision guards remain authoritative.
    """
    import mujoco

    if (
        type(ball_geom_id) is not int
        or not 0 <= ball_geom_id < model.ngeom
        or len(leg_dof_ids) != 2
        or any(len(ids) != 6 for ids in leg_dof_ids)
        or any(dof < 0 or dof >= model.nv for ids in leg_dof_ids for dof in ids)
    ):
        raise ValueError("valid ball and paired six-leg DoF mapping required")
    gaps: list[float] = []
    gradients: list[tuple[float, ...]] = []
    prefix = agent_id.replace(".", "_")
    for side, dofs in zip(("left", "right"), leg_dof_ids, strict=True):
        shin_geom_id = model.geom(f"{prefix}_{side}_shin").id
        closest: NDArray[np.float64] = np.zeros(6, dtype=np.float64)
        gap = float(mujoco.mj_geomDistance(model, data, shin_geom_id, ball_geom_id, 100.0, closest))
        direction = closest[3:] - closest[:3]
        direction_norm = float(np.linalg.norm(direction))
        if not math.isfinite(gap) or direction_norm <= 1e-9:
            raise ValueError("degenerate measured shin-ball closest points")
        linear = np.zeros((3, model.nv), dtype=np.float64)
        angular = np.zeros((3, model.nv), dtype=np.float64)
        body_id = int(model.geom_bodyid[shin_geom_id])
        mujoco.mj_jac(model, data, linear, angular, closest[:3], body_id)
        gradient = -(direction / direction_norm) @ linear[:, list(dofs)]
        gaps.append(gap)
        gradients.append(tuple(float(value) for value in gradient))
    result = TeamShinClearance(
        agent_id,
        frame,
        (gaps[0], gaps[1]),
        (gradients[0], gradients[1]),
    )
    result.__post_init__()
    return result
