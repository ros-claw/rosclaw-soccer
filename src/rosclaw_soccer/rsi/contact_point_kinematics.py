"""SIM-only readback of instantaneous body velocities at a reconstructed contact."""

import mujoco
import numpy as np


def contact_point_relative_velocity(
    model, data, point, normal, ball_body: int, player_body: int
) -> dict:
    """Velocity of ball minus player at one world contact point; no force estimate."""
    point, normal = np.asarray(point), np.asarray(normal)
    if (
        point.shape != (3,)
        or normal.shape != (3,)
        or not np.isfinite(point).all()
        or not np.isfinite(normal).all()
        or not np.isclose(np.linalg.norm(normal), 1.0, rtol=0, atol=1e-5)
        or type(ball_body) is not int
        or type(player_body) is not int
        or not 0 < ball_body < model.nbody
        or not 0 < player_body < model.nbody
        or ball_body == player_body
        or not np.isfinite(data.qvel).all()
    ):
        raise ValueError("finite world contact point, normal and distinct physical bodies required")
    ball_jac = np.zeros((3, model.nv))
    player_jac = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, ball_jac, None, point, ball_body)
    mujoco.mj_jac(model, data, player_jac, None, point, player_body)
    ball = ball_jac @ data.qvel
    player = player_jac @ data.qvel
    relative = ball - player
    signed_normal = float(relative @ normal)
    tangent = relative - signed_normal * normal
    return dict(
        ball_velocity_world_mps=ball.tolist(),
        player_velocity_world_mps=player.tolist(),
        relative_velocity_world_mps=relative.tolist(),
        relative_speed_mps=float(np.linalg.norm(relative)),
        normal_speed_abs_mps=abs(signed_normal),
        tangent_speed_mps=float(np.linalg.norm(tangent)),
        force_reconstructed=False,
    )
