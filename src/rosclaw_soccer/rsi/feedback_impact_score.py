"""Complete first-impact-window proxy on owned feedback forecasts, not M0 truth.

Reject predictions with an incomplete20ms window after first foot contact.
No simulator, neural model, training, control or policy promotion runs here.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.physics.g1_feedback_forecast import G1FeedbackForecast
from rosclaw_soccer.rsi.impact_torque_screen import CAPS


def feedback_impact_score(forecast: G1FeedbackForecast, offset_nm: np.ndarray) -> dict[str, Any]:
    if type(forecast) is not G1FeedbackForecast:
        raise ValueError("owned full-feedback40ms physical prediction required")
    if (
        not isinstance(offset_nm, np.ndarray)
        or offset_nm.shape != (29,)
        or offset_nm.dtype != np.float64
        or not np.isfinite(offset_nm).all()
        or np.any(abs(offset_nm) > CAPS)
    ):
        raise ValueError("bounded canonical torque action required")
    a = forecast.arrays
    for name, shape, dtype in (
        ("canonical_qvel_after_substep", (20, 41), np.float64),
        ("pelvis_height_solver_epoch_m", (20,), np.float64),
        ("robot_force_on_ball_world_n", (20, 3), np.float64),
        ("foot_contact_peak_force_n_after_substep", (20, 2), np.float64),
        ("completed_foot_ground_normal_n", (20, 2), np.float64),
        ("nonfoot_robot_contact_over1n_after_substep", (20,), np.bool_),
    ):
        value = a.get(name)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != dtype
            or not np.isfinite(value).all()
            or np.max(abs(value)) > 1e6
        ):
            raise ValueError(
                "whole finite typed physical feedback/contact/support predictions required"
            )
    if np.any(a["foot_contact_peak_force_n_after_substep"] < 0) or np.any(
        a["completed_foot_ground_normal_n"] < 0
    ):
        raise ValueError("nonnegative force magnitudes required")
    touched = np.flatnonzero(np.any(a["foot_contact_peak_force_n_after_substep"] > 1, axis=1))
    first = None if not len(touched) else int(touched[0])
    complete = first is not None and first + 10 <= 20
    nonfoot = bool(np.any(a["nonfoot_robot_contact_over1n_after_substep"]))
    height = float(a["pelvis_height_solver_epoch_m"].min())
    safe = not nonfoot and height >= 0.65
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
    penalty = 1e-4 * float(np.sum((offset_nm / CAPS) ** 2))
    if complete and first is not None:
        velocity = a["canonical_qvel_after_substep"][first + 9, 35:38]
        impulse = a["robot_force_on_ball_world_n"][first : first + 10].sum(axis=0) * 0.002
        vx, vy, _ = (float(v) for v in velocity)
        if safe:
            cost = max(abs(vy) - 0.25 * max(vx, 0), 0) ** 2 + max(1 - vx, 0) ** 2 + penalty
    return dict(
        schema="soccer.rsi.full_feedback_initial_contact_proxy.v1",
        prediction_horizon_substeps=20,
        prediction_dt_s=0.002,
        first_predicted_foot_substep=first,
        initial_contact_window_substeps=10,
        complete_initial_contact_window_observed=bool(complete),
        feasible_predicted_impact=bool(safe and complete),
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
        post40ms_stability_proven=False,
        neural_policy_trained=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
