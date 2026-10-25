"""Offline geometric witnesses at a recorded native force clock, no integration.

Contacts cached by mj_step precede that step's stored post-integration qpos.
We reconstruct geometry only; never claim to reproduce solved contact forces.
"""

from typing import Any

import numpy as np


def inspect_ball_geometry(
    model: Any,
    qpos: np.ndarray,
    *,
    native_index: int,
    robot_prefix: str = "a/",
    ball_body_name: str = "soccer_ball",
) -> dict[str, Any]:
    import mujoco

    if (
        type(native_index) is not int
        or not 1 <= native_index < len(qpos)
        or (
            qpos.ndim != 2
            or qpos.shape[1] != model.nq
            or qpos.dtype != np.float64
            or not np.isfinite(qpos).all()
        )
    ):
        raise ValueError("actual force index with available preceding native state required")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos[native_index - 1]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    ball = int(model.body(ball_body_name).id)
    root = int(model.body(robot_prefix + "pelvis").id)
    feet = {
        int(model.body(robot_prefix + name).id)
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    }
    witnesses = []
    for contact in data.contact[: data.ncon]:
        g1, g2 = map(int, contact.geom)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        other, geom = (b2, g2) if b1 == ball else (b1, g1) if b2 == ball else (-1, -1)
        if other < 0:
            continue
        ancestor = other
        while ancestor not in (0, root):
            ancestor = int(model.body_parentid[ancestor])
        if ancestor != root:
            continue
        witnesses.append(
            dict(
                body=model.body(other).name,
                geom=model.geom(geom).name or f"geom-{geom}",
                is_allowed_foot=other in feet,
                signed_distance_m=float(contact.dist),
                contact_position=contact.pos.tolist(),
                geom_type=int(model.geom_type[geom]),
            )
        )
    if data.time != 0:
        raise ValueError("geometric inspection advanced the physical clock")
    return dict(
        force_native_index=native_index,
        inspected_preintegration_qpos_index=native_index - 1,
        geometric_contacts=witnesses,
        solved_contact_forces_reconstructed=False,
        new_physical_steps=0,
        model_clock_advanced=False,
    )
