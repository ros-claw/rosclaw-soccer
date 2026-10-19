"""Post-touch exit proxy, NOT an M0 label or full-body teacher qualification.

Unlike the precontact initial20ms objective, scores the end of the full80ms
branch and remaining foot contact after the60ms action has ended. A measured
completed foot-contact bin is mandatory; no new predicted impact is required.
Weights are fixed research parameters, never task-success thresholds.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.physics.g1_early_feedback_forecast import G1EarlyFeedbackForecast
from rosclaw_soccer.rsi.early_feedback_impact_score import early_feedback_impact_score


def postcontact_exit_score(
    forecast: G1EarlyFeedbackForecast,
    offset_nm: np.ndarray,
    *,
    previous_contact_forces: np.ndarray,
) -> dict[str, Any]:
    if (
        type(previous_contact_forces) is not np.ndarray
        or previous_contact_forces.shape != (6,)
        or previous_contact_forces.dtype != np.float64
        or not np.isfinite(previous_contact_forces).all()
        or np.any(previous_contact_forces < 0)
        or not np.any(previous_contact_forces[:2] > 1)
    ):
        raise ValueError("actual completed typed foot-contact forces required for exit proxy")
    # Reuse ONLY the existing strict raw80ms array/action validation and shared
    # diagnostics, never its precontact feasibility, initial velocity or cost.
    checked = early_feedback_impact_score(forecast, offset_nm)
    a = forecast.arrays
    velocity = a["canonical_qvel_after_substep"][-1, 35:38]
    vx, vy, _ = (float(v) for v in velocity)
    contact = np.any(a["foot_contact_peak_force_n_after_substep"][30:] > 1, axis=1)
    contact_fraction = float(contact.mean())
    negative_forward_impulse = float(
        np.maximum(-a["robot_force_on_ball_world_n"][:, 0], 0).sum() * 0.002
    )
    reason = (
        "OBSERVED_PRIOR_NONFOOT_CONTACT"
        if np.any(previous_contact_forces[2:] > 1)
        else "PREDICTED_NONFOOT_CONTACT"
        if checked["predicted_nonfoot_contact_over1n"]
        else "PREDICTED_LOW_PELVIS"
        if checked["minimum_pelvis_z_m"] < 0.65
        else None
    )
    speed_cost = max(1.2 - vx, 0) ** 2
    lateral_cost = max(abs(vy) - 0.25 * max(vx, 0), 0) ** 2
    held_contact_cost = 0.2 * contact_fraction**2
    drag_cost = 0.1 * negative_forward_impulse**2
    cost = speed_cost + lateral_cost + held_contact_cost + drag_cost + checked["action_regularizer"]
    return dict(
        schema="soccer.rsi.postcontact80ms_exit_proxy.v1",
        prediction_horizon_substeps=40,
        prediction_dt_s=0.002,
        action_duration_substeps=30,
        feedback_target_updates=3,
        completed_contact_side=int(np.argmax(previous_contact_forces[:2])),
        feasible_predicted_exit=reason is None,
        rejection_reason=reason,
        terminal_ball_velocity_m_s=velocity.tolist(),
        terminal_washout_foot_contact_fraction=contact_fraction,
        negative_forward_robot_impulse_n_s=negative_forward_impulse,
        speed_cost=speed_cost,
        lateral_cost=lateral_cost,
        held_contact_cost=held_contact_cost,
        drag_cost=drag_cost,
        action_regularizer=checked["action_regularizer"],
        minimum_pelvis_z_m=checked["minimum_pelvis_z_m"],
        predicted_nonfoot_contact_over1n=checked["predicted_nonfoot_contact_over1n"],
        maximum_root_angular_speed_rad_s=checked["maximum_root_angular_speed_rad_s"],
        proxy_cost=None if reason is not None else cost,
        new_predicted_foot_contact_required=False,
        final_m0_success_proven=False,
        post80ms_stability_proven=False,
        neural_policy_trained=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
