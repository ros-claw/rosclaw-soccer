"""Bounded static SIM reference retargeting experiment; not a motor controller."""

from typing import Any, cast

import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash


def fit_reference_support_pose(
    model: Any,
    root_pose_wxyz: np.ndarray,
    joints_hg29: np.ndarray,
    claimed_contacts: np.ndarray,
    *,
    expected_model_hash: str,
    maximum_evaluations: int = 2000,
) -> dict[str, Any]:
    """Locally fit labeled feet to floor, retaining pose and physical joint bounds.

    This operates on one scratch pose and never integrates. A numerically
    feasible result is an unqualified candidate reference, not physical motion.
    """
    import mujoco
    from scipy.optimize import least_squares

    if (
        compiled_model_hash(model) != expected_model_hash
        or type(root_pose_wxyz) is not np.ndarray
        or root_pose_wxyz.shape != (7,)
        or root_pose_wxyz.dtype.kind != "f"
        or not np.isfinite(root_pose_wxyz).all()
        or abs(np.linalg.norm(root_pose_wxyz[3:]) - 1) > 1e-4
        or type(joints_hg29) is not np.ndarray
        or joints_hg29.shape != (29,)
        or joints_hg29.dtype.kind != "f"
        or not np.isfinite(joints_hg29).all()
        or type(claimed_contacts) is not np.ndarray
        or claimed_contacts.shape != (2,)
        or claimed_contacts.dtype != np.bool_
        or not claimed_contacts.any()
        or type(maximum_evaluations) is not int
        or not 14 <= maximum_evaluations <= 2000
    ):
        raise ValueError("pinned body and finite pose with explicit claimed supports required")
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    floor = int(model.geom("floor").id)
    feet = [
        [int(model.geom(f"a/{side}_foot{i}_collision").id) for i in range(1, 8)]
        for side in ("left", "right")
    ]
    if (
        model.geom_type[floor] != int(mujoco.mjtGeom.mjGEOM_PLANE)
        or model.geom_bodyid[floor] != 0
        or any(
            model.geom_type[g] != int(mujoco.mjtGeom.mjGEOM_CAPSULE) for foot in feet for g in foot
        )
    ):
        raise ValueError("fixed plane and expected capsule foot geometry required")
    limits = model.jnt_range[layout.joints]
    limited = model.jnt_limited[layout.joints].astype(bool)
    q0 = joints_hg29.astype(np.float64, copy=True)
    root0 = root_pose_wxyz.astype(np.float64, copy=True)
    if np.any(limited & ((q0 < limits[:, 0]) | (q0 > limits[:, 1]))):
        raise ValueError("source joints must already satisfy physical position limits")
    leg_low = np.maximum(-0.2, limits[:12, 0] - q0[:12])
    leg_high = np.minimum(0.2, limits[:12, 1] - q0[:12])
    if not limited[:12].all() or np.any(leg_low >= leg_high):
        raise ValueError("twelve bounded legal leg joints required")
    lower = np.r_[-0.15, leg_low]
    upper = np.r_[0.15, leg_high]
    data = mujoco.MjData(model)
    data.qpos[layout.bq : layout.bq + 7] = [100, 100, 0.11, 1, 0, 0, 0]
    calls = 0

    def geometry(x: np.ndarray) -> np.ndarray:
        data.qpos[layout.rq : layout.rq + 7] = root0
        data.qpos[layout.rq + 2] += x[0]
        data.qpos[layout.qi] = q0
        data.qpos[layout.qi[:12]] = q0[:12] + x[1:]
        mujoco.mj_kinematics(model, data)
        if not np.allclose(data.geom_xmat[floor].reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-9):
            raise ValueError("upward fixed floor plane required")
        floor_z = data.geom_xpos[floor, 2]
        return np.array(
            [
                min(
                    float(
                        data.geom_xpos[g, 2]
                        - model.geom_size[g, 0]
                        - model.geom_size[g, 1] * abs(data.geom_xmat[g].reshape(3, 3)[2, 2])
                        - floor_z
                    )
                    for g in foot
                )
                for foot in feet
            ]
        )

    def residual(x: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls > maximum_evaluations:
            raise ValueError("bounded reference IK evaluation budget exhausted")
        gap = geometry(x)
        return cast(np.ndarray, np.r_[100 * gap[claimed_contacts], 100 * np.minimum(gap, 0), 2 * x])

    before = geometry(np.zeros(13))
    result = least_squares(
        residual, np.zeros(13), bounds=(lower, upper), max_nfev=100, ftol=1e-9, xtol=1e-9, gtol=1e-9
    )
    after = geometry(result.x)
    candidate_q = q0.copy()
    candidate_q[:12] += result.x[1:]
    candidate_root = root0.copy()
    candidate_root[2] += result.x[0]
    feasible = bool(
        result.success
        and np.max(np.abs(after[claimed_contacts])) <= 0.01
        and np.min(after) >= -0.002
        and candidate_root[2] >= 0.65
        and np.isfinite(candidate_q).all()
        and data.time == 0
    )
    return dict(
        root_pose_wxyz=candidate_root,
        joints_hg29=candidate_q,
        original_foot_gaps_m=before,
        candidate_foot_gaps_m=after,
        root_z_change_m=float(result.x[0]),
        maximum_leg_change_rad=float(np.max(np.abs(result.x[1:]))),
        joint_trust_boundary_active=bool(np.any(np.abs(result.x[1:]) >= 0.2 - 1e-6)),
        root_trust_boundary_active=bool(abs(result.x[0]) >= 0.15 - 1e-6),
        solver_evaluations=calls,
        solver_success=bool(result.success),
        reference_geometry_feasible=feasible,
        source_contact_labels_assumed=True,
        input_mapping_and_provenance_assumed=True,
        ankle_xy_and_temporal_continuity_not_constrained=True,
        nonfoot_collision_not_qualified=True,
        integrated_native_steps=0,
        physical_execution_qualified=False,
        hardware_authorized=False,
    )
