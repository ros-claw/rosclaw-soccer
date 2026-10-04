"""Measured collision-point geometry for football response prediction."""

from typing import Any

import numpy as np


def measured_foot_ball_contact_features(model: Any, current_data: Any) -> np.ndarray[Any, Any]:
    """Read a caller-owned, CURRENT forwarded simulation state, without stepping.

    Eight original foot collision spheres, left then right, model geom order:
    ball-minus-sphere position (3), ball-minus-point velocity (3), surface gap
    (1). Geometry and point Jacobians are measured, not future collision labels.
    Visual meshes are explicitly excluded. This does not change collision sizes.
    """
    import mujoco

    ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    if ball < 0 or ball_geom < 0 or model.geom_type[ball_geom] != mujoco.mjtGeom.mjGEOM_SPHERE:
        raise ValueError("original measured sphere ball required")
    ball_joint = int(model.body_jntadr[ball])
    if ball_joint < 0 or model.jnt_type[ball_joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("free ball velocity reference required")
    ball_velocity = current_data.qvel[int(model.jnt_dofadr[ball_joint]) :][:3]
    ball_position = current_data.geom_xpos[ball_geom]
    features = []
    for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            raise ValueError("complete measured foot geometry required")
        geometries = np.flatnonzero(
            (model.geom_bodyid == body)
            & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
            & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        )
        if len(geometries) != 4:
            raise ValueError("four original collision spheres per foot required")
        for geometry in geometries:
            point = current_data.geom_xpos[geometry]
            jacobian = np.empty((3, model.nv))
            mujoco.mj_jac(model, current_data, jacobian, None, point, body)
            relative_position = ball_position - point
            relative_velocity = ball_velocity - jacobian @ current_data.qvel
            gap = float(
                np.linalg.norm(relative_position)
                - model.geom_size[ball_geom, 0]
                - model.geom_size[geometry, 0]
            )
            features.append(np.concatenate((relative_position, relative_velocity, [gap])))
    result: np.ndarray[Any, Any] = np.asarray(features, dtype=np.float32).reshape(56)
    if not np.isfinite(result).all() or np.max(np.abs(result)) > 1e6:
        raise ValueError("finite current collision-point features required")
    result.flags.writeable = False
    return result
