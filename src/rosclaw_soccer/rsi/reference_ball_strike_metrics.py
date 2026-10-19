"""Contact-complete SIM diagnostics, never an official football exam gate."""

from typing import Any

import numpy as np


def summarize_reference_ball_strike(model: Any, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    contact = arrays["ball_contact_force_by_body_n"]
    q = arrays["target_body_canonical_qpos_after_substep"]
    v = arrays["target_body_canonical_qvel_after_substep"]
    if (
        contact.shape != (300, 10, model.nbody)
        or q.shape != (300, 10, 43)
        or v.shape != (300, 10, 41)
        or any(not np.isfinite(a).all() for a in (contact, q, v))
        or np.any(contact < 0)
    ):
        raise ValueError("complete finite native300x10 contact/position/velocity required")
    pelvis = int(model.body("pelvis").id)
    root = model.body_rootid[pelvis]
    if root == 0:
        raise ValueError("independent robot body root required")
    feet = [
        int(model.body(side + suffix).id)
        for side in ("left", "right")
        for suffix in ("_ankle_roll_link", "_ankle_pitch_link")
    ]
    robot = np.flatnonzero(model.body_rootid == root).tolist()
    nonfeet = [i for i in robot if i not in feet]
    if not set(feet).issubset(robot) or not nonfeet:
        raise ValueError("named feet and all nonfoot robot bodies required")
    f = contact.reshape(3000, model.nbody)

    def first(indices: list[int]) -> int | None:
        where = np.flatnonzero(np.any(f[:, indices] > 1.0, axis=1))
        return int(where[0]) if len(where) else None

    return dict(
        schema="soccer.rsi.reference_ball_strike_diagnostic.v1",
        first_foot_contact_native_index=first(feet),
        first_nonfoot_robot_contact_native_index=first(nonfeet),
        maximum_foot_force_n=float(f[:, feet].max()),
        maximum_nonfoot_robot_force_n=float(f[:, nonfeet].max()),
        nonfoot_includes_all_robot_bodies_not_only_knees=True,
        final_ball_displacement_m=(q[-1, -1, 36:39] - arrays["canonical_qpos"][0, 36:39]).tolist(),
        maximum_ball_center_height_m=float(q[:, :, 38].max()),
        maximum_ball_freejoint_forward_velocity_mps=float(v[:, :, 35].max()),
        velocity_convention="postintegration_canonical_freejoint_not_cached_object_velocity",
        contact_native_index_convention="zero_based_after_step_contact_solver_epoch",
        official_football_exam_pass=False,
        promotion_authorized=False,
    )
