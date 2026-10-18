"""World-frame spherical-ball surface velocity from current MuJoCo state."""

from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.native_ball_dimensions import inspect_native_ball_dimensions


def inspect_ball_surface_velocity(
    model: mujoco.MjModel, data: mujoco.MjData, *, geom_id: int
) -> dict[str, Any]:
    """Update kinematics only; no integration, contact solve or state mutation.

    Bottom-point velocity is a hypothetical surface-point velocity even when the
    ball is airborne. It is NOT measured ground contact or friction impulse.
    """
    ball = inspect_native_ball_dimensions(model, geom_id=geom_id)
    if not np.array_equal(model.geom_pos[geom_id], np.zeros(3)) or not np.array_equal(
        model.body_ipos[ball.body_id], np.zeros(3)
    ):
        raise ValueError("sphere centered at its body center of mass required")
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise ValueError("finite current native state required")
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_comVel(model, data)
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, ball.body_id, velocity, 0)
    angular, linear = velocity[:3], velocity[3:]
    surface = linear + np.cross(angular, [0, 0, -ball.radius_m])
    return dict(
        ball_radius_m=ball.radius_m,
        center_position_world_m=data.geom_xpos[geom_id].tolist(),
        angular_velocity_world_rad_s=angular.tolist(),
        center_velocity_world_m_s=linear.tolist(),
        hypothetical_bottom_point_velocity_world_m_s=surface.tolist(),
        hypothetical_bottom_horizontal_speed_m_s=float(np.linalg.norm(surface[:2])),
        ground_contact_established_by_this_function=False,
        new_integration_steps=0,
    )
