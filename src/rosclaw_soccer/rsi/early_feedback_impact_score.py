"""80ms/60ms-action impact proxy, never a full task or teacher qualification."""

from typing import Any

import numpy as np

from rosclaw_soccer.physics.g1_early_feedback_forecast import G1EarlyFeedbackForecast
from rosclaw_soccer.rsi.impact_torque_screen import CAPS


def early_feedback_impact_score(
    forecast: G1EarlyFeedbackForecast, offset_nm: np.ndarray
) -> dict[str, Any]:
    if type(forecast) is not G1EarlyFeedbackForecast:
        raise ValueError("owned80ms causal feedback prediction required")
    if (
        not isinstance(offset_nm, np.ndarray)
        or offset_nm.shape != (29,)
        or offset_nm.dtype != np.float64
        or not np.isfinite(offset_nm).all()
        or np.any(abs(offset_nm) > CAPS)
    ):
        raise ValueError("bounded canonical60ms torque correction required")
    a = forecast.arrays
    for name, shape, dtype in (
        ("canonical_qvel_after_substep", (40, 41), np.float64),
        ("pelvis_height_solver_epoch_m", (40,), np.float64),
        ("robot_force_on_ball_world_n", (40, 3), np.float64),
        ("foot_contact_peak_force_n_after_substep", (40, 2), np.float64),
        ("completed_foot_ground_normal_n", (40, 2), np.float64),
        ("nonfoot_robot_contact_over1n_after_substep", (40,), np.bool_),
        ("requested_offset_nm", (40, 29), np.float64),
    ):
        v = a.get(name)
        if (
            not isinstance(v, np.ndarray)
            or v.shape != shape
            or v.dtype != dtype
            or not np.isfinite(v).all()
            or np.max(abs(v)) > 1e6
        ):
            raise ValueError("whole finite typed80ms dynamics/contact/action required")
    if (
        np.any(a["foot_contact_peak_force_n_after_substep"] < 0)
        or np.any(a["completed_foot_ground_normal_n"] < 0)
        or not np.array_equal(a["requested_offset_nm"][:30], np.tile(offset_nm, (30, 1)))
        or np.any(a["requested_offset_nm"][30:])
    ):
        raise ValueError("nonnegative contact magnitudes and exact60ms-then-zero action required")
    touched = np.flatnonzero(np.any(a["foot_contact_peak_force_n_after_substep"] > 1, axis=1))
    first = None if not len(touched) else int(touched[0])
    complete = first is not None and first + 10 <= 40
    nonfoot = bool(np.any(a["nonfoot_robot_contact_over1n_after_substep"]))
    height = float(a["pelvis_height_solver_epoch_m"].min())
    reason = (
        "PREDICTED_NONFOOT_CONTACT"
        if nonfoot
        else "PREDICTED_LOW_PELVIS"
        if height < 0.65
        else "NO_PREDICTED_FOOT_CONTACT"
        if first is None
        else "RIGHT_CENSORED_INITIAL_CONTACT_WINDOW"
        if not complete
        else None
    )
    velocity, impulse, cost = None, None, None
    # Same normalized action penalty per control period, integrated over3 periods.
    penalty = 3e-4 * float(np.sum((offset_nm / CAPS) ** 2))
    if complete and first is not None:
        velocity = a["canonical_qvel_after_substep"][first + 9, 35:38]
        impulse = a["robot_force_on_ball_world_n"][first : first + 10].sum(axis=0) * 0.002
        vx, vy, _ = (float(v) for v in velocity)
        if reason is None:
            cost = max(abs(vy) - 0.25 * max(vx, 0), 0) ** 2 + max(1 - vx, 0) ** 2 + penalty
    return dict(
        schema="soccer.rsi.early80ms_initial_contact_proxy.v1",
        prediction_horizon_substeps=40,
        prediction_dt_s=0.002,
        action_duration_substeps=30,
        feedback_target_updates=3,
        first_predicted_foot_substep=first,
        initial_contact_window_substeps=10,
        complete_initial_contact_window_observed=bool(complete),
        feasible_predicted_impact=reason is None,
        rejection_reason=reason,
        initial_window_end_ball_velocity_m_s=None if velocity is None else velocity.tolist(),
        initial_window_robot_impulse_n_s=None if impulse is None else impulse.tolist(),
        minimum_pelvis_z_m=height,
        predicted_nonfoot_contact_over1n=nonfoot,
        peak_foot_ground_normal_n=a["completed_foot_ground_normal_n"].max(axis=0).tolist(),
        maximum_root_angular_speed_rad_s=float(
            np.linalg.norm(a["canonical_qvel_after_substep"][:, 3:6], axis=1).max()
        ),
        action_regularizer=penalty,
        proxy_cost=cost,
        final_m0_success_proven=False,
        post80ms_stability_proven=False,
        neural_policy_trained=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
