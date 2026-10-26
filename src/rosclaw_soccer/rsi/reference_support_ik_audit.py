"""Independent static geometric readback for support-IK reference candidates."""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash


def audit_support_candidate(
    model: Any,
    source_root: np.ndarray,
    source_joints: np.ndarray,
    candidate_root: np.ndarray,
    candidate_joints: np.ndarray,
    claimed_contacts: np.ndarray,
    *,
    expected_model_hash: str,
    max_claimed_ankle_xy_change_m: float | None = None,
) -> dict[str, Any]:
    """Use MuJoCo's geometry-distance query, without replaying the optimizer."""
    import mujoco

    arrays = (source_root, source_joints, candidate_root, candidate_joints, claimed_contacts)
    if (
        compiled_model_hash(model) != expected_model_hash
        or any(type(x) is not np.ndarray for x in arrays)
        or source_root.shape != (7,)
        or candidate_root.shape != (7,)
        or source_joints.shape != (29,)
        or candidate_joints.shape != (29,)
        or claimed_contacts.shape != (2,)
        or claimed_contacts.dtype != np.bool_
        or not claimed_contacts.any()
        or any(not np.isfinite(x).all() for x in arrays)
        or abs(np.linalg.norm(source_root[3:]) - 1) > 1e-4
        or not np.array_equal(source_root[[0, 1, 3, 4, 5, 6]], candidate_root[[0, 1, 3, 4, 5, 6]])
        or not np.array_equal(source_joints[12:], candidate_joints[12:])
        or (
            max_claimed_ankle_xy_change_m is not None
            and (
                type(max_claimed_ankle_xy_change_m) is not float
                or not np.isfinite(max_claimed_ankle_xy_change_m)
                or not 0 < max_claimed_ankle_xy_change_m <= 0.02
            )
        )
    ):
        raise ValueError(
            "exact pinned body, bounded pose fields and unchanged non-leg reference required"
        )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    limits = model.jnt_range[layout.joints]
    limited = model.jnt_limited[layout.joints].astype(bool)
    root_delta = float(candidate_root[2] - source_root[2])
    leg_delta = candidate_joints[:12] - source_joints[:12]
    if (
        abs(root_delta) > 0.15 + 1e-9
        or np.max(np.abs(leg_delta)) > 0.2 + 1e-9
        or np.any(
            limited
            & ((candidate_joints < limits[:, 0] - 1e-9) | (candidate_joints > limits[:, 1] + 1e-9))
        )
    ):
        raise ValueError("candidate exceeds declared root, leg or physical joint limits")
    floor = int(model.geom("floor").id)
    feet = [
        [int(model.geom(f"a/{side}_foot{i}_collision").id) for i in range(1, 8)]
        for side in ("left", "right")
    ]
    if model.geom_type[floor] != int(mujoco.mjtGeom.mjGEOM_PLANE) or any(
        model.geom_type[g] != int(mujoco.mjtGeom.mjGEOM_CAPSULE) for foot in feet for g in foot
    ):
        raise ValueError("expected floor plane and capsule feet required")
    foot_geoms = {g for foot in feet for g in foot}
    ankle_ids = [int(model.body(f"a/{side}_ankle_roll_link").id) for side in ("left", "right")]
    data = mujoco.MjData(model)
    data.qpos[layout.bq : layout.bq + 7] = [100, 100, 0.11, 1, 0, 0, 0]

    def measure(root: np.ndarray, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        data.qpos[layout.rq : layout.rq + 7] = root
        data.qpos[layout.qi] = joints
        mujoco.mj_forward(model, data)
        gaps = np.array(
            [
                min(float(mujoco.mj_geomDistance(model, data, floor, g, 1.0, None)) for g in foot)
                for foot in feet
            ]
        )
        nonfoot_floor = [
            float(contact.dist)
            for contact in data.contact
            if floor in (int(contact.geom[0]), int(contact.geom[1]))
            and not ({int(contact.geom[0]), int(contact.geom[1])} & foot_geoms)
        ]
        penetration = min(nonfoot_floor, default=0.0)
        return gaps, data.xpos[ankle_ids].copy(), penetration

    before, ankle_before, nonfoot_before = measure(source_root, source_joints)
    after, ankle_after, nonfoot_after = measure(candidate_root, candidate_joints)
    if data.time != 0:
        raise ValueError("static geometric readback cannot advance physics")
    ankle_changes = np.linalg.norm(ankle_after[:, :2] - ankle_before[:, :2], axis=1)
    feasible = bool(
        np.max(np.abs(after[claimed_contacts])) <= 0.01
        and np.min(after) >= -0.002
        and candidate_root[2] >= 0.65
        and (
            max_claimed_ankle_xy_change_m is None
            or np.max(ankle_changes[claimed_contacts]) <= max_claimed_ankle_xy_change_m
        )
    )
    return dict(
        source_foot_gaps_m=before.tolist(),
        candidate_foot_gaps_m=after.tolist(),
        max_claimed_foot_error_m=float(np.max(np.abs(after[claimed_contacts]))),
        max_ankle_xy_displacement_m=float(np.max(ankle_changes)),
        max_claimed_ankle_xy_displacement_m=float(np.max(ankle_changes[claimed_contacts])),
        claimed_ankle_xy_limit_m=max_claimed_ankle_xy_change_m,
        source_minimum_nonfoot_floor_distance_m=nonfoot_before,
        candidate_minimum_nonfoot_floor_distance_m=nonfoot_after,
        root_z_change_m=root_delta,
        maximum_leg_change_rad=float(np.max(np.abs(leg_delta))),
        leg_trust_boundary_active=bool(np.any(np.abs(leg_delta) >= 0.2 - 1e-6)),
        reference_geometry_feasible=feasible,
        other_contacts_not_in_feasibility=True,
        continuous_motion_not_assessed=True,
        source_contact_labels_assumed=True,
        input_mapping_and_provenance_assumed=True,
        integrated_native_steps=0,
        physical_execution_qualified=False,
        hardware_authorized=False,
    )
