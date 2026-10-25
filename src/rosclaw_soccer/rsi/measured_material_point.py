"""Read-only current-state material-point Jacobians for qualified SIM layouts.

No integration, inverse dynamics, root/ball authority, or contact-force claim.
The local material point stays fixed to its body during differentiation.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout


def measure_material_point(
    model: Any,
    donor: Any,
    layout: NativeVelocityLayout,
    *,
    body_name: str,
    local_point: np.ndarray,
) -> dict[str, Any]:
    import mujoco

    if (
        layout.model is not model
        or not isinstance(local_point, np.ndarray)
        or local_point.shape != (3,)
        or local_point.dtype != np.float64
        or not np.isfinite(local_point).all()
        or donor.qpos.shape != (model.nq,)
        or not np.isfinite(donor.qpos).all()
        or not np.isfinite(donor.qvel).all()
    ):
        raise ValueError("qualified own layout and finite actual state/material point required")
    body = int(model.body(body_name).id)
    ancestor = body
    while ancestor not in (0, layout.pelvis):
        ancestor = int(model.body_parentid[ancestor])
    if ancestor != layout.pelvis:
        raise ValueError("material point must belong to the selected player's body")
    layout.canonical(donor)
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, donor)
    # mj_step caches transforms before integration. Recompute from actual latest qpos.
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_comPos(model, scratch)
    point = scratch.xpos[body] + scratch.xmat[body].reshape(3, 3) @ local_point
    jac = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, scratch, jac, None, point, body)
    if not np.isfinite(jac).all():
        raise ValueError("nonfinite measured material-point Jacobian")
    return dict(
        body=body_name,
        local_point=local_point.copy(),
        world_point=point.copy(),
        joint_jacobian_hg=jac[:, layout.vi].copy(),
        joint_qpos_hg=donor.qpos[layout.qi].copy(),
        joint_dof_ids_hg=layout.vi.copy(),
        actual_time=float(donor.time),
        new_physical_steps=0,
        floating_root_and_ball_columns_excluded=True,
        hardware_authorized=False,
    )
