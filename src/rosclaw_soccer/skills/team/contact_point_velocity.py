"""Read-only MuJoCo world-point relative velocity for ball contact evidence."""

from __future__ import annotations

from typing import Any

import numpy as np


def measure_contact_relative_velocity_world_mps(
    *,
    model: Any,
    data: Any,
    ball_geom: int,
    counterpart_geom: int,
    contact_position_world_m: np.ndarray[Any, Any],
) -> tuple[float, float, float]:
    """Return counterpart velocity minus ball velocity at the same world point."""
    import mujoco

    point = np.asarray(contact_position_world_m, dtype=float)
    velocity = np.asarray(data.qvel, dtype=float)
    if (
        point.shape != (3,)
        or velocity.shape != (model.nv,)
        or not np.isfinite(point).all()
        or not np.isfinite(velocity).all()
        or min(ball_geom, counterpart_geom) < 0
        or max(ball_geom, counterpart_geom) >= model.ngeom
    ):
        raise ValueError("finite measured contact point and geometry IDs required")
    linear = np.zeros((3, model.nv))
    angular = np.zeros((3, model.nv))
    ball_body = int(model.geom_bodyid[ball_geom])
    other_body = int(model.geom_bodyid[counterpart_geom])
    mujoco.mj_jac(model, data, linear, angular, point, ball_body)
    ball_velocity = linear @ velocity
    mujoco.mj_jac(model, data, linear, angular, point, other_body)
    relative = linear @ velocity - ball_velocity
    if not np.isfinite(relative).all() or np.max(np.abs(relative)) > 1000:
        raise ValueError("nonfinite or unbounded physical contact-relative velocity")
    return float(relative[0]), float(relative[1]), float(relative[2])
