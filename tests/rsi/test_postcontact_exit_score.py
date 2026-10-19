"""Synthetic exit-objective checks; do not imply a physically good teacher."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.early_feedback_impact_score import early_feedback_impact_score
from rosclaw_soccer.rsi.postcontact_exit_score import postcontact_exit_score
from tests.rsi.test_early_feedback_impact_score import fixture


def score(forecast, offset, forces=None):
    return postcontact_exit_score(
        forecast,
        offset,
        previous_contact_forces=np.array([2.0, 0, 0, 0, 0, 0]) if forces is None else forces,
    )


def test_actual_posttouch_can_release_without_predicting_another_impact():
    f, offset = fixture(first=None)
    assert not early_feedback_impact_score(f, offset)["feasible_predicted_impact"]
    s = score(f, offset)
    assert s["feasible_predicted_exit"]
    assert s["terminal_washout_foot_contact_fraction"] == 0
    assert s["new_predicted_foot_contact_required"] is False
    assert s["completed_contact_side"] == 0
    for key in (
        "final_m0_success_proven",
        "post80ms_stability_proven",
        "teacher_qualified",
        "neural_policy_trained",
        "hardware_authorized",
        "promotion_authorized",
    ):
        assert s[key] is False


def test_speed_is_final80ms_not_the_initial_contact_proxy():
    f, offset = fixture(first=0)
    f.arrays["canonical_qvel_after_substep"][9, 35:38] = [1.5, 0, 0]
    f.arrays["canonical_qvel_after_substep"][-1, 35:38] = [0.3, 0, 0]
    assert early_feedback_impact_score(f, offset)["proxy_cost"] == 0
    bad = score(f, offset)
    assert bad["terminal_ball_velocity_m_s"] == [0.3, 0, 0]
    assert bad["speed_cost"] == pytest.approx(0.81)
    f.arrays["canonical_qvel_after_substep"][-1, 35:38] = [1.2, 0.2, 0]
    assert score(f, offset)["proxy_cost"] < bad["proxy_cost"]


def test_terminal_contact_and_backward_robot_impulse_are_penalized():
    f, offset = fixture(first=None)
    f.arrays["canonical_qvel_after_substep"][:, 35:38] = [1.2, 0, 0]
    good = score(f, offset)
    f.arrays["foot_contact_peak_force_n_after_substep"][30:, 0] = 2
    f.arrays["robot_force_on_ball_world_n"][:, 0] = -10
    bad = score(f, offset)
    assert bad["terminal_washout_foot_contact_fraction"] == 1
    assert bad["negative_forward_robot_impulse_n_s"] == pytest.approx(0.8)
    assert bad["held_contact_cost"] == pytest.approx(0.2)
    assert bad["proxy_cost"] > good["proxy_cost"]


@pytest.mark.parametrize("step", [0, 10, 20, 30, 39])
@pytest.mark.parametrize("bad", ["nonfoot", "height"])
def test_any_predicted_unsafe_substep_rejects_whole_exit(step, bad):
    f, offset = fixture(first=None)
    if bad == "nonfoot":
        f.arrays["nonfoot_robot_contact_over1n_after_substep"][step] = True
    else:
        f.arrays["pelvis_height_solver_epoch_m"][step] = 0.649
    s = score(f, offset)
    assert not s["feasible_predicted_exit"] and s["proxy_cost"] is None


@pytest.mark.parametrize("bad", ["none", "weak", "negative", "nan", "shape", "dtype"])
def test_observed_foot_contact_cannot_be_invented(bad):
    f, offset = fixture()
    forces = np.array([2.0, 0, 0, 0, 0, 0])
    if bad == "none":
        forces[:] = 0
    elif bad == "weak":
        forces[0] = 1
    elif bad == "negative":
        forces[1] = -1
    elif bad == "nan":
        forces[5] = np.nan
    elif bad == "shape":
        forces = forces[:2]
    else:
        forces = forces.astype(np.float32)
    with pytest.raises(ValueError, match="completed typed foot-contact"):
        score(f, offset, forces)


def test_prior_nonfoot_observation_rejects_even_good_prediction():
    f, offset = fixture(first=None)
    s = score(f, offset, np.array([0.0, 2, 0, 0, 2, 0]))
    assert s["completed_contact_side"] == 1
    assert s["rejection_reason"] == "OBSERVED_PRIOR_NONFOOT_CONTACT"
    assert s["proxy_cost"] is None


def test_whole_raw_array_and_action_validation_is_not_bypassed():
    f, offset = fixture(first=None)
    f.arrays["canonical_qvel_after_substep"] = f.arrays["canonical_qvel_after_substep"][:-1]
    with pytest.raises(ValueError):
        score(f, offset)
