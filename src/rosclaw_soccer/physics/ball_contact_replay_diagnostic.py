"""Read contact geometry/forces using the collision-stage velocity snapshot.

Call after a single Euler mj_step with its saved BEFORE-step qvel. Contact
positions/Jacobians describe that collision stage, not the integrated qpos.
No stepping, forward refresh, model change or controller is performed here.
"""

from typing import Any

import numpy as np


def ball_contact_samples(
    model: Any, data: Any, ball_geom: int, velocity_before_step: np.ndarray[Any, Any]
) -> list[dict[str, Any]]:
    import mujoco

    if (
        type(ball_geom) is not int
        or not 0 <= ball_geom < model.ngeom
        or model.geom_type[ball_geom] != mujoco.mjtGeom.mjGEOM_SPHERE
        or model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER
        or velocity_before_step.shape != (model.nv,)
        or velocity_before_step.dtype != np.float64
        or not np.isfinite(velocity_before_step).all()
    ):
        raise ValueError("explicit sphere, Euler collision stage and finite saved qvel required")
    result = []
    for index in range(data.ncon):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if ball_geom not in (g1, g2):
            continue
        other = g2 if g1 == ball_geom else g1
        if other < 0:
            raise ValueError("rigid geometry contact diagnostic does not accept flex contacts")
        frame = np.asarray(contact.frame).reshape(3, 3)
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, force)
        sign = 1.0 if g2 == ball_geom else -1.0
        normal = sign * frame[0]
        point = np.asarray(contact.pos).copy()
        ball_jac = np.zeros((3, model.nv))
        other_jac = np.zeros_like(ball_jac)
        ball_body, other_body = int(model.geom_bodyid[ball_geom]), int(model.geom_bodyid[other])
        mujoco.mj_jac(model, data, ball_jac, None, point, ball_body)
        mujoco.mj_jac(model, data, other_jac, None, point, other_body)
        ball_velocity, other_velocity = (
            ball_jac @ velocity_before_step,
            other_jac @ velocity_before_step,
        )
        relative = ball_velocity - other_velocity
        force_world = sign * (frame.T @ force[:3])
        if not all(np.isfinite(a).all() for a in (frame, force, point, relative, force_world)):
            raise ValueError("finite original contact state required")
        result.append(
            dict(
                geom1=g1,
                geom2=g2,
                other_geom=other,
                other_geom_name=model.geom(other).name,
                other_body=other_body,
                other_body_name=model.body(other_body).name,
                point_world_m=point.tolist(),
                normal_other_to_ball_world=normal.tolist(),
                contact_force_components_n=force[:3].tolist(),
                force_on_ball_world_n=force_world.tolist(),
                force_norm_n=float(np.linalg.norm(force[:3])),
                distance_m=float(contact.dist),
                ball_point_velocity_world_m_s=ball_velocity.tolist(),
                other_point_velocity_world_m_s=other_velocity.tolist(),
                ball_minus_other_point_velocity_world_m_s=relative.tolist(),
                signed_normal_relative_velocity_m_s=float(normal @ relative),
            )
        )
    return result
