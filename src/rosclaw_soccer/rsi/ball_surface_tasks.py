"""Current measured foot/shin-to-ball surfaces, no integration or forces.

Only smooth separated pairs expose a gap gradient. Penetration/degeneracy
remains measured evidence, never a fabricated differentiable clearance.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout


def measure_ball_surfaces(model: Any, donor: Any, layout: NativeVelocityLayout) -> dict[str, Any]:
    import mujoco

    if layout.model is not model:
        raise ValueError("own qualified layout required")
    layout.canonical(donor)
    ball = np.flatnonzero(model.geom_bodyid == layout.ball)
    if len(ball) != 1 or model.geom_type[ball[0]] != mujoco.mjtGeom.mjGEOM_SPHERE:
        raise ValueError("single actual sphere required")
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, donor)
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_comPos(model, scratch)
    records: dict[str, Any] = {}
    for side in ("left", "right"):
        for part, suffix in (("shin", "knee_link"), ("foot", "ankle_roll_link")):
            body = int(model.body(layout.robot_prefix + side + "_" + suffix).id)
            geoms = np.flatnonzero(
                (model.geom_bodyid == body)
                & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
            )
            if part == "shin":
                geoms = np.array([int(model.geom(layout.robot_prefix + side + "_shin").id)])
                if int(model.geom_bodyid[geoms[0]]) != body:
                    raise ValueError("shin collision surface belongs to unexpected body")
            candidates = []
            for geom in geoms:
                segment = np.zeros(6, np.float64)
                gap = float(
                    mujoco.mj_geomDistance(model, scratch, int(geom), int(ball[0]), 2.0, segment)
                )
                if np.isfinite(gap) and gap < 2.0:
                    candidates.append((gap, int(geom), segment.copy()))
            if not candidates:
                records[side + "_" + part] = dict(available=False)
                continue
            gap, selected_geom, segment = min(candidates, key=lambda item: item[0])
            jac = np.zeros((3, model.nv), np.float64)
            mujoco.mj_jac(model, scratch, jac, None, segment[:3], body)
            smooth = gap > 1e-6 and abs(np.linalg.norm(segment[3:] - segment[:3]) - gap) < 1e-7
            normal = (segment[3:] - segment[:3]) / gap if smooth else None
            records[side + "_" + part] = dict(
                available=True,
                geom=selected_geom,
                body=body,
                gap_m=gap,
                world_point=segment[:3],
                ball_point=segment[3:],
                local_point=scratch.xmat[body].reshape(3, 3).T @ (segment[:3] - scratch.xpos[body]),
                jacobian_hg=jac[:, layout.vi].copy(),
                point_velocity=jac @ donor.qvel,
                normal_toward_ball=normal,
                gap_jacobian_hg=None if normal is None else -normal @ jac[:, layout.vi],
                smooth_separated=smooth,
            )
    return dict(surfaces=records, actual_time=float(donor.time), new_physical_steps=0)
