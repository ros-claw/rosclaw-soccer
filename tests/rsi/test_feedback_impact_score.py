"""Complete-window proxy arithmetic only; not a real G1 policy success."""

import numpy as np
import pytest

from rosclaw_soccer.physics.g1_feedback_forecast import G1FeedbackForecast
from rosclaw_soccer.rsi.feedback_impact_score import feedback_impact_score


def prediction(first=9):
    a = dict(
        canonical_qvel_after_substep=np.zeros((20, 41)),
        pelvis_height_solver_epoch_m=np.full(20, 0.7),
        robot_force_on_ball_world_n=np.zeros((20, 3)),
        foot_contact_peak_force_n_after_substep=np.zeros((20, 2)),
        completed_foot_ground_normal_n=np.full((20, 2), 100.0),
        nonfoot_robot_contact_over1n_after_substep=np.zeros(20, dtype=np.bool_),
    )
    a["canonical_qvel_after_substep"][:, 35] = 2
    if first is not None:
        a["foot_contact_peak_force_n_after_substep"][first:, 0] = 10
        a["robot_force_on_ball_world_n"][first:, 0] = 10
    return G1FeedbackForecast("synthetic", np.zeros(1), a, {})


def test_contact_at_last_substep_of_first_control_still_has_complete_window():
    f = prediction(9)
    a = feedback_impact_score(f, np.zeros(29))
    assert a["first_predicted_foot_substep"] == 9 and a["complete_initial_contact_window_observed"]
    assert a["feasible_predicted_impact"] and a["proxy_cost"] == 0
    assert a["initial_window_robot_impulse_n_s"] == [0.2, 0, 0]
    assert a["initial_window_end_ball_velocity_m_s"] == [2, 0, 0]


def test_uses_exact_first_contact_window_not_arbitrary_control_end():
    f = prediction(9)
    f.arrays["canonical_qvel_after_substep"][18, 36] = 1
    f.arrays["canonical_qvel_after_substep"][19, 35:38] = [9, 0, 0]
    result = feedback_impact_score(f, np.zeros(29))
    assert result["initial_window_end_ball_velocity_m_s"] == [2, 1, 0]
    assert result["proxy_cost"] == 0.25


@pytest.mark.parametrize("first", [None, 11, 19])
def test_no_invented_or_clipped_initial_contact_window(first):
    result = feedback_impact_score(prediction(first), np.zeros(29))
    assert not result["feasible_predicted_impact"]
    assert result["proxy_cost"] is None and result["initial_window_end_ball_velocity_m_s"] is None
    assert result["rejection_reason"] in (
        "NO_PREDICTED_FOOT_CONTACT",
        "RIGHT_CENSORED_INITIAL_CONTACT_WINDOW",
    )


@pytest.mark.parametrize("bad", ["nonfoot", "height"])
def test_local_safety_veto_cannot_win(bad):
    f = prediction(0)
    if bad == "nonfoot":
        f.arrays["nonfoot_robot_contact_over1n_after_substep"][19] = True
    else:
        f.arrays["pelvis_height_solver_epoch_m"][19] = 0.649
    r = feedback_impact_score(f, np.zeros(29))
    assert r["complete_initial_contact_window_observed"] and not r["feasible_predicted_impact"]
    assert r["proxy_cost"] is None


@pytest.mark.parametrize("bad", ["missing", "nan", "huge", "bool_dtype", "negative", "offset"])
def test_unknown_malformed_or_unbounded_predictions_rejected(bad):
    f = prediction()
    offset = np.zeros(29)
    if bad == "missing":
        del f.arrays["nonfoot_robot_contact_over1n_after_substep"]
    elif bad == "nan":
        f.arrays["canonical_qvel_after_substep"][19, 35] = np.nan
    elif bad == "huge":
        f.arrays["canonical_qvel_after_substep"][19, 35] = 1e200
    elif bad == "bool_dtype":
        f.arrays["nonfoot_robot_contact_over1n_after_substep"] = np.zeros(20)
    elif bad == "negative":
        f.arrays["completed_foot_ground_normal_n"][0, 0] = -1
    else:
        offset[0] = 100
    with pytest.raises(ValueError):
        feedback_impact_score(f, offset)


def test_proxy_never_promotes_or_claims_final_task_stability_or_learning():
    r = feedback_impact_score(prediction(), np.zeros(29))
    for name in (
        "final_m0_success_proven",
        "post40ms_stability_proven",
        "neural_policy_trained",
        "teacher_qualified",
        "hardware_authorized",
        "promotion_authorized",
    ):
        assert r[name] is False
