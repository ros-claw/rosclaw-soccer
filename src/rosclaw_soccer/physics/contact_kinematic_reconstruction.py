"""Read-only scratch contact kinematics, never integration or measured force.

Recorded post-step states must be shifted to the pre-step solver clock by the
caller. A geometric contact does NOT identify which pair carried recorded force.
"""

from typing import Any

import numpy as np


def native_pre_step_state(
    initial_qpos: np.ndarray,
    initial_qvel: np.ndarray,
    post_qpos: np.ndarray,
    post_qvel: np.ndarray,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        type(index) is not int
        or initial_qpos.ndim != 1
        or initial_qvel.ndim != 1
        or post_qpos.ndim != 2
        or post_qvel.ndim != 2
        or post_qpos.shape[1:] != initial_qpos.shape
        or post_qvel.shape[1:] != initial_qvel.shape
        or len(post_qpos) != len(post_qvel)
        or not 0 <= index < len(post_qpos)
        or any(x.dtype != np.float64 for x in (initial_qpos, initial_qvel, post_qpos, post_qvel))
    ):
        raise ValueError("exact float64 initial/post-step trace and native index required")
    q, v = (
        (initial_qpos, initial_qvel) if index == 0 else (post_qpos[index - 1], post_qvel[index - 1])
    )
    if not np.isfinite(q).all() or not np.isfinite(v).all():
        raise ValueError("finite pre-step configuration and velocity required")
    return q.copy(), v.copy()


def contact_kinematics(
    model: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    ball_body: int,
    contact_bodies: tuple[int, ...],
    goal: np.ndarray,
) -> list[dict[str, Any]]:
    """Compile no world, step no physics, read no force from the scratch solve."""
    import mujoco

    if (
        mujoco.get_mjcb_control() is not None
        or model.nplugin
        or qpos.shape != (model.nq,)
        or qvel.shape != (model.nv,)
        or goal.shape != (3,)
        or any(x.dtype != np.float64 or not np.isfinite(x).all() for x in (qpos, qvel, goal))
        or type(ball_body) is not int
        or not 0 < ball_body < model.nbody
        or type(contact_bodies) is not tuple
        or not contact_bodies
        or len(set(contact_bodies)) != len(contact_bodies)
        or any(
            type(b) is not int or not 0 < b < model.nbody or b == ball_body for b in contact_bodies
        )
    ):
        raise ValueError("explicit callback-free scratch sphere/body contact state required")
    geoms = np.flatnonzero(model.geom_bodyid == ball_body)
    if len(geoms) != 1 or model.geom_type[geoms[0]] != int(mujoco.mjtGeom.mjGEOM_SPHERE):
        raise ValueError("one actual sphere ball required for radial contact interpretation")
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = qpos, qvel
    mujoco.mj_forward(model, data)
    result = []
    for c in data.contact:
        g1, g2 = map(int, c.geom)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        body = b2 if b1 == ball_body else b1 if b2 == ball_body else -1
        if body not in contact_bodies:
            continue
        point = c.pos.copy()
        radial = data.xpos[ball_body] - point
        normal = c.frame.reshape(3, 3)[0].copy()
        if radial @ normal < 0:
            normal *= -1  # Explicit convention towards sphere center, not a force measurement.
        if np.linalg.norm(radial) < 1e-8 or not np.isclose(np.linalg.norm(normal), 1, atol=1e-8):
            raise ValueError("nondegenerate actual sphere contact geometry required")
        velocities = []
        for b in (body, ball_body):
            jac = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jac, None, point, b)
            velocities.append(jac @ qvel)
        foot_v, ball_v = velocities
        heading = goal[:2] - data.xpos[ball_body, :2]
        angle = None
        if np.linalg.norm(heading) > 1e-8 and np.linalg.norm(normal[:2]) > 1e-8:
            angle = float(
                np.degrees(
                    np.arctan2(
                        heading[0] * normal[1] - heading[1] * normal[0], heading @ normal[:2]
                    )
                )
            )
        foot_geom = g2 if b1 == ball_body else g1
        result.append(
            dict(
                body=model.body(body).name,
                geom=model.geom(foot_geom).name,
                geom_type=int(model.geom_type[foot_geom]),
                distance_m=float(c.dist),
                contact_point_world=point.tolist(),
                ball_center_world=data.xpos[ball_body].tolist(),
                point_body_local=(
                    data.xmat[body].reshape(3, 3).T @ (point - data.xpos[body])
                ).tolist(),
                normal_towards_ball=normal.tolist(),
                goal_heading_error_degrees=angle,
                body_point_velocity_world=foot_v.tolist(),
                ball_point_velocity_world=ball_v.tolist(),
                closing_normal_velocity_mps=float((foot_v - ball_v) @ normal),
                reconstructed_constraint_included=bool(c.efc_address >= 0),
                geometric_pair_is_recorded_force_source=False,
                scratch_force_used=False,
            )
        )
    return result
