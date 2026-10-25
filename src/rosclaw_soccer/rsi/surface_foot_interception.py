"""Measured foot-surface interception, distinct from ankle-to-ball-center pursuit.

SIM privileged teacher only. Geometry is observed, never moved. The target is
the actual sphere surface on the foot-to-ball ray, with no tuned padding.
"""

import math
from typing import Any

import numpy as np


def foot_surface_points(model: Any, snapshot: Any, layout: Any) -> np.ndarray:
    import mujoco

    if layout.model is not model or snapshot.qpos.shape != (model.nq,):
        raise ValueError("own current canonical model/snapshot required")
    ball_geoms = np.flatnonzero(model.geom_bodyid == layout.ball)
    if (
        len(ball_geoms) != 1
        or model.geom_type[ball_geoms[0]] != mujoco.mjtGeom.mjGEOM_SPHERE
        or abs(model.geom_size[ball_geoms[0], 0] - 0.11) > 1e-12
    ):
        raise ValueError("actual standard sphere required for surface target")
    points = []
    for side in ("left", "right"):
        body = model.body(layout.robot_prefix + side + "_ankle_roll_link").id
        geoms = np.flatnonzero(
            (model.geom_bodyid == body)
            & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        )
        candidates = []
        for geom in geoms:
            segment = np.zeros(6)
            gap = float(mujoco.mj_geomDistance(model, snapshot, geom, ball_geoms[0], 2.0, segment))
            if gap < 2.0:
                candidates.append((gap, segment[:3].copy()))
        if not candidates:
            raise ValueError("near-ball physical foot geometry unavailable; no fabricated point")
        points.append(min(candidates, key=lambda x: x[0])[1])
    return np.stack(points)


def surface_foot_interception_command(
    *,
    yaw: float,
    surfaces: np.ndarray,
    ball_position: np.ndarray,
    ball_velocity_xy: np.ndarray,
    previous_nominal_world_command: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    from rosclaw_soccer.skills.planar_interception import (
        PlanarInterceptionConfig,
        propose_planar_interception,
    )

    if (
        not math.isfinite(yaw)
        or any(
            not isinstance(a, np.ndarray) or a.shape != shape or not np.isfinite(a).all()
            for a, shape in (
                (surfaces, (2, 3)),
                (ball_position, (3,)),
                (ball_velocity_xy, (2,)),
                (previous_nominal_world_command, (2,)),
            )
        )
        or np.linalg.norm(previous_nominal_world_command) > 1 + 1e-6
    ):
        raise ValueError("finite measured foot surfaces, actual ball and nominal history required")
    foot = int(np.argmin(np.linalg.norm(surfaces - ball_position, axis=1)))
    point = surfaces[foot].copy()
    ray = ball_position - point
    length = float(np.linalg.norm(ray))
    degenerate = length < 1e-12
    offset = np.zeros(3) if degenerate else -0.11 * ray / length
    target = ball_position + offset
    proposed = propose_planar_interception(
        target_position_xy=torch.tensor(ball_position[:2], dtype=torch.float32),
        target_velocity_xy=torch.tensor(ball_velocity_xy, dtype=torch.float32),
        effector_position_xy=torch.tensor(point[:2], dtype=torch.float32),
        previous_velocity_xy=torch.tensor(previous_nominal_world_command, dtype=torch.float32),
        config=PlanarInterceptionConfig((float(offset[0]), float(offset[1])), 1.5, 1.0, 0.08),
    ).numpy()
    c, s = math.cos(yaw), math.sin(yaw)
    desired = math.atan2(float(proposed[1]), float(proposed[0]))
    error = math.atan2(math.sin(desired - yaw), math.cos(desired - yaw))
    command = np.array(
        [
            np.clip(c * proposed[0] + s * proposed[1], -0.5, 1),
            np.clip(-s * proposed[0] + c * proposed[1], -0.5, 0.5),
            np.clip(2 * error, -1, 1),
        ],
        np.float32,
    )
    return command, dict(
        active=True,
        selected_foot=foot,
        point=point,
        target=target,
        degenerate_ray=degenerate,
    )
