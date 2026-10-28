"""Bounded, offline reference pose fitting; never a runtime motion controller.

The optimizer operates on scratch forward kinematics only. Its acceptance
criteria concern surface geometry, not force, trajectory, balance or control.
"""

from typing import Any

import numpy as np
from scipy.optimize import least_squares

from rosclaw_soccer.rsi.ball_surface_tasks import measure_ball_surfaces


def fit_reference_contact(model: Any, donor: Any, layout: Any, *, foot: str) -> dict[str, Any]:
    """Fit one 29-DOF reference frame while preserving the other ankle pose.

    A 0.35 rad per-joint trust region and native model limits apply to the
    first 12 HG leg joints. No source, live simulator, or ball state is edited.
    """
    import mujoco

    if layout.model is not model or foot not in ("left", "right"):
        raise ValueError("qualified model and explicit foot required")
    layout.canonical(donor)
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, donor)
    mujoco.mj_kinematics(model, scratch)
    mujoco.mj_comPos(model, scratch)
    support = "right" if foot == "left" else "left"
    body = int(model.body(layout.robot_prefix + support + "_ankle_roll_link").id)
    anchor_pos = scratch.xpos[body].copy()
    anchor_rot = scratch.xmat[body].reshape(3, 3).copy()
    original = scratch.qpos[layout.qi].copy()
    limits = model.jnt_range[layout.joints[:12]].astype(np.float64)
    if not np.all(model.jnt_limited[layout.joints[:12]]):
        raise ValueError("native leg joint limits required")
    lower = np.maximum(-0.35, limits[:, 0] - original[:12])
    upper = np.minimum(0.35, limits[:, 1] - original[:12])
    if np.any(lower > 0) or np.any(upper < 0) or np.any(lower >= upper):
        return dict(
            status="SOURCE_OUTSIDE_NATIVE_LIMITS",
            physically_qualified=False,
            new_physical_steps=0,
            hardware_authorized=False,
        )

    def metrics(delta: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        scratch.qpos[layout.qi[:12]] = original[:12] + delta
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_comPos(model, scratch)
        surfaces = measure_ball_surfaces(model, scratch, layout)["surfaces"]
        if not all(
            surfaces[name]["available"] for name in (foot + "_foot", "left_shin", "right_shin")
        ):
            raise ValueError("actual foot and both shin surfaces required")
        pos = scratch.xpos[body] - anchor_pos
        rel = anchor_rot.T @ scratch.xmat[body].reshape(3, 3)
        # Full rotation matrix residual remains valid near 180-degree rotations.
        rot = (rel - np.eye(3)).reshape(-1)
        angle = float(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)))
        values = dict(
            foot_gap_m=float(surfaces[foot + "_foot"]["gap_m"]),
            left_shin_gap_m=float(surfaces["left_shin"]["gap_m"]),
            right_shin_gap_m=float(surfaces["right_shin"]["gap_m"]),
            support_position_error_m=float(np.linalg.norm(pos)),
            support_angle_error_rad=angle,
        )
        residual = np.concatenate(
            (
                np.array([(values["foot_gap_m"] - 0.004) * 100]),
                np.array(
                    [
                        max(0.0, 0.015 - values[k]) * 100
                        for k in ("left_shin_gap_m", "right_shin_gap_m")
                    ]
                ),
                pos * 100,
                rot * 5,
                delta * 0.2,
            )
        )
        return values, residual

    before, _ = metrics(np.zeros(12))
    result = least_squares(
        lambda x: metrics(x)[1],
        np.zeros(12),
        bounds=(lower, upper),
        max_nfev=150,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    after, _ = metrics(result.x)
    feasible = bool(
        result.success
        and -0.002 <= after["foot_gap_m"] <= 0.01
        and min(after["left_shin_gap_m"], after["right_shin_gap_m"]) >= 0.01
        and after["support_position_error_m"] <= 0.01
        and after["support_angle_error_rad"] <= 0.05
    )
    return dict(
        status="STATIC_GEOMETRY_FEASIBLE" if feasible else "STATIC_GEOMETRY_NOT_QUALIFIED",
        foot=foot,
        before=before,
        after=after,
        delta_leg_rad=result.x.tolist(),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        evaluations=int(result.nfev),
        physically_qualified=False,
        new_physical_steps=0,
        hardware_authorized=False,
    )
