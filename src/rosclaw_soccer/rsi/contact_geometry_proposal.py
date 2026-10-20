"""Own-forecast geometry proposal only; no stepping, targets or qualification.

Caller must bind full300 rows to its own measured prefix + native forecast.
An archive or FK residual is not dynamic reachability or an imitation label.
"""

from typing import Any

import numpy as np


def contact_geometry_proposal(
    qpos: np.ndarray,
    qvel: np.ndarray,
    toe_world: np.ndarray,
    nonfoot_force: np.ndarray,
    *,
    handoff_frame: int | None,
    ball_radius_m: float,
) -> dict[str, Any]:
    """One deterministic frame/vector, fixed bounds; never consumes task labels."""
    if (
        any(
            type(a) is not np.ndarray
            or a.shape != shape
            or a.dtype != np.float64
            or not np.isfinite(a).all()
            for a, shape in (
                (qpos, (300, 43)),
                (qvel, (300, 41)),
                (toe_world, (300, 3)),
                (nonfoot_force, (300, 10)),
            )
        )
        or np.any(nonfoot_force < 0)
        or not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, rtol=0, atol=1e-5)
        or ball_radius_m != 0.11
        or (
            handoff_frame is not None
            and (type(handoff_frame) is not int or not 30 <= handoff_frame <= 90)
        )
    ):
        raise ValueError("finite full300 own-forecast geometry and original ball/entry required")
    report: dict[str, Any] = dict(
        schema="soccer.rsi.contact_geometry_proposal.v1",
        applicable=False,
        offset_root_m=None,
        selected_global_frame=None,
        landmark="front_two_5mm_support_sphere_midpoint_not_mesh",
        native_steps=0,
        neural_inferences=0,
        optimizer_updates=0,
        physics_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if handoff_frame is None:
        return dict(report, reason="no_original_handoff")
    # Central FK derivative is a diagnostic, not contact impulse or native velocity.
    velocity = (toe_world[2:] - toe_world[:-2]) / 0.04
    candidates = []
    for frame in range(handoff_frame + 91, min(handoff_frame + 210, 299)):
        local = frame - handoff_frame
        weight = float(np.sin(np.pi * (local - 90) / 120) ** 2)
        if weight <= 0.1 or velocity[frame - 1, 0] - qvel[frame, 35] <= 0.2:
            continue
        desired = qpos[frame, 36:39] - np.array([ball_radius_m + 0.005, 0, 0])
        residual = desired - toe_world[frame]
        distance = float(np.linalg.norm(residual))
        if not np.isfinite(distance):
            raise ValueError("derived contact geometry must remain finite")
        candidates.append((distance, frame, weight, residual))
    if not candidates:
        return dict(report, reason="no_forward_approach_in_fixed_contact_window")
    distance, frame, weight, residual = min(candidates, key=lambda x: (x[0], x[1]))
    if np.any(nonfoot_force[:frame] > 1.0):
        return dict(
            report, reason="nonfoot_ball_contact_before_selected_frame", selected_global_frame=frame
        )
    # Scalar-first unit quaternion, rotate world residual into actual root frame.
    w, x, y, z = qpos[frame, 3:7]
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    request = rotation.T @ residual / weight
    if not np.isfinite(request).all() or not np.isfinite(np.linalg.norm(request)):
        raise ValueError("derived contact geometry must remain finite")
    scale = min(
        1.0,
        0.04 / max(float(np.max(abs(request))), 1e-12),
        0.06 / max(float(np.linalg.norm(request)), 1e-12),
    )
    offset = request * scale
    # Round inward only if floating point multiplication crosses a declared bound.
    while np.max(abs(offset)) > 0.04 or np.linalg.norm(offset) > 0.06:
        scale = float(np.nextafter(scale, 0.0))
        offset = request * scale
    return dict(
        report,
        applicable=True,
        reason="one_bounded_geometric_proposal_not_contact_proof",
        selected_global_frame=frame,
        selected_reference_local_frame=frame - handoff_frame,
        temporal_weight=weight,
        root_local_unbounded_request_m=request.tolist(),
        proposal_trust_region_scale=scale,
        offset_root_m=offset.tolist(),
        world_residual_m=residual.tolist(),
        predicted_toe_to_back_surface_distance_m=distance,
        relative_forward_approach_mps=float(velocity[frame - 1, 0] - qvel[frame, 35]),
        proposal_projected=scale < 1,
    )


def forecast_contact_geometry(
    model: Any, arrays: dict[str, np.ndarray], *, handoff_frame: int | None
) -> dict[str, Any]:
    """Private MjData FK of recorded own forecast; never touch actual live data."""
    import mujoco

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES

    if (model.nq, model.nv, model.nu) != (43, 41, 29) or model.opt.timestep != 0.002:
        raise ValueError("original G1 native model required")
    q = arrays["canonical_qpos"]
    v = arrays["canonical_qvel"]
    contact = arrays["ball_contact_force_by_body_n"]
    if (
        q.shape != (300, 43)
        or v.shape != (300, 41)
        or contact.shape != (300, 10, model.nbody)
        or any(a.dtype != np.float64 or not np.isfinite(a).all() for a in (q, v, contact))
    ):
        raise ValueError("complete finite full300 own recorded trajectory required")
    ball = int(model.body("ball").id)
    ball_geoms = np.flatnonzero(model.geom_bodyid == ball)
    if (
        len(ball_geoms) != 1
        or model.geom_type[ball_geoms[0]] != mujoco.mjtGeom.mjGEOM_SPHERE
        or model.geom_size[ball_geoms[0], 0] != 0.11
        or model.body_mass[ball] != 0.43
    ):
        raise ValueError("original single sphere ball dimensions required")
    foot = int(model.body("right_ankle_roll_link").id)
    spheres = np.flatnonzero(
        (model.geom_bodyid == foot)
        & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
        & (model.geom_contype != 0)
    )
    if len(spheres) != 4 or not np.all(model.geom_size[spheres, 0] == 0.005):
        raise ValueError("original four right-foot5mm supports required")
    toes = spheres[model.geom_pos[spheres, 0] == model.geom_pos[spheres, 0].max()]
    if len(toes) != 2:
        raise ValueError("exact original two front supports required")
    joints = [model.joint(n).id for n in G1_DDS_JOINT_NAMES]
    qi = model.jnt_qposadr[joints]
    scratch = mujoco.MjData(model)
    points = np.empty((300, 3))
    for i, row in enumerate(q):
        scratch.qpos[:7] = row[:7]
        scratch.qpos[qi] = row[7:36]
        mujoco.mj_kinematics(model, scratch)
        points[i] = scratch.geom_xpos[toes].mean(axis=0)
    root = model.body_rootid[model.body("pelvis").id]
    feet = {
        int(model.body(side + suffix).id)
        for side in ("left", "right")
        for suffix in ("_ankle_roll_link", "_ankle_pitch_link")
    }
    nonfeet = [i for i in range(model.nbody) if model.body_rootid[i] == root and i not in feet]
    if root == 0 or not nonfeet:
        raise ValueError("independent named robot root required")
    return contact_geometry_proposal(
        q,
        v,
        points,
        contact[:, :, nonfeet].max(axis=2),
        handoff_frame=handoff_frame,
        ball_radius_m=0.11,
    )
