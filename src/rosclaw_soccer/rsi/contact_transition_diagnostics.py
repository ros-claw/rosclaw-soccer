"""Offline contact timing and surface geometry; never reconstruct unsaved contact forces."""

from typing import Any

import numpy as np


def diagnose_contact_transition(
    model: Any, layout: Any, arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    import mujoco

    q, v, forces = (arrays[k] for k in ("qpos", "qvel", "ball_forces"))
    if (
        layout.model is not model
        or model.opt.timestep != 0.002
        or q.ndim != 2
        or not 1 <= len(q) <= 3600
        or q.shape[1] != model.nq
        or v.shape != (len(q), model.nv)
        or forces.shape != (len(q), 1, 3)
        or any(x.dtype != np.float64 or not np.isfinite(x).all() for x in (q, v, forces))
        or (forces < 0).any()
    ):
        raise ValueError(
            "complete finite single-player500Hz native state and contact evidence required"
        )
    formal = min(len(q), 3000)
    feet = np.flatnonzero((forces[:formal, 0, :2] > 1).any(axis=1))
    nonfeet = np.flatnonzero(forces[:formal, 0, 2] > 1)
    first = int(feet[0]) if len(feet) else None
    nonfoot = int(nonfeet[0]) if len(nonfeet) else None
    event_frames = {}
    if first is not None:
        for name, offset in (
            ("before_foot_40ms", -20),
            ("before_foot_10ms", -5),
            ("first_foot", 0),
        ):
            if first + offset >= 0:
                event_frames[name] = first + offset
    if nonfoot is not None:
        event_frames["first_nonfoot"] = nonfoot
    ball_geoms = np.flatnonzero(model.geom_bodyid == layout.ball)
    if len(ball_geoms) != 1 or model.geom_type[ball_geoms[0]] != mujoco.mjtGeom.mjGEOM_SPHERE:
        raise ValueError("one actual native ball sphere required")
    foot_bodies = {
        int(model.body(layout.robot_prefix + side + "_ankle_roll_link").id)
        for side in ("left", "right")
    }
    robot_geoms = []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        ancestor = body
        while ancestor not in (0, layout.pelvis):
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == layout.pelvis and (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            robot_geoms.append(geom)
    d = mujoco.MjData(model)
    states = []
    for name, n in event_frames.items():
        # Saved force[n] was measured at the PRE-integration collision geometry;
        # saved q[n] is POST-integration. Never conflate these two clocks.
        if n == 0:
            states.append(
                dict(event=name, native_index=n, geometry_status="INITIAL_STATE_NOT_SAVED")
            )
            continue
        d.qpos[:], d.qvel[:] = q[n - 1], v[n - 1]
        mujoco.mj_kinematics(model, d)
        gaps = []
        for geom in robot_geoms:
            segment = np.zeros(6)
            gap = float(mujoco.mj_geomDistance(model, d, geom, int(ball_geoms[0]), 2.0, segment))
            if gap < 2.0:
                body = int(model.geom_bodyid[geom])
                gaps.append(
                    dict(
                        geom=model.geom(geom).name,
                        body=model.body(body).name,
                        is_foot=body in foot_bodies,
                        gap_m=gap,
                    )
                )
        states.append(
            dict(
                event=name,
                native_index=n,
                control_frame=n // 10,
                geometry_state_native_index=n - 1,
                geometry_status="OBSERVED_PRE_INTEGRATION",
                ball_position_m=d.qpos[layout.bq : layout.bq + 3].tolist(),
                ball_velocity_m_s=d.qvel[layout.bv : layout.bv + 3].tolist(),
                nearest_foot_surfaces=sorted(
                    (g for g in gaps if g["is_foot"]), key=lambda g: g["gap_m"]
                )[:4],
                nearest_nonfoot_surfaces=sorted(
                    (g for g in gaps if not g["is_foot"]), key=lambda g: g["gap_m"]
                )[:4],
                observed_force_channels_n=forces[n, 0].tolist(),
            )
        )
    if float(d.time) != 0:
        raise ValueError("offline geometry unexpectedly advanced physical time")
    return dict(
        first_foot_native_index=first,
        first_nonfoot_native_index=nonfoot,
        foot_to_nonfoot_ms=None if first is None or nonfoot is None else 2 * (nonfoot - first),
        observed_nonfoot_samples=len(nonfeet),
        observed_nonfoot_peak_n=float(forces[:formal, 0, 2].max()),
        states=states,
        collision_body_force_identity_reconstructed=False,
        new_physical_steps=0,
        neural_optimizer_updates=0,
        hardware_authorized=False,
    )
