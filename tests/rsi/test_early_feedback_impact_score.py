"""Synthetic80ms proxy examples; no real policy trials or trained teacher."""

import numpy as np
import pytest

from rosclaw_soccer.physics.g1_early_feedback_forecast import G1EarlyFeedbackForecast
from rosclaw_soccer.rsi.early_feedback_impact_score import early_feedback_impact_score
from rosclaw_soccer.rsi.impact_torque_screen import CAPS


def fixture(*, first=29, offset=None):
    offset = np.zeros(29) if offset is None else offset
    a = dict(
        canonical_qvel_after_substep=np.zeros((40, 41)),
        pelvis_height_solver_epoch_m=np.full(40, 0.7),
        robot_force_on_ball_world_n=np.zeros((40, 3)),
        foot_contact_peak_force_n_after_substep=np.zeros((40, 2)),
        completed_foot_ground_normal_n=np.zeros((40, 2)),
        nonfoot_robot_contact_over1n_after_substep=np.zeros(40, dtype=np.bool_),
        requested_offset_nm=np.vstack((np.tile(offset, (30, 1)), np.zeros((10, 29)))),
    )
    a["canonical_qvel_after_substep"][:, 35:38] = [2.0, 0.9, 0.0]
    if first is not None:
        a["foot_contact_peak_force_n_after_substep"][first, 0] = 2.0
        a["robot_force_on_ball_world_n"][first : first + 10] = [10.0, 2.0, 0.0]
    return G1EarlyFeedbackForecast("sha256:" + "a" * 64, np.zeros(100), a, (), ()), offset


def test_complete_impact_proxy_uses_full_first20ms_not_a_single_contact_substep():
    forecast, offset = fixture()
    s = early_feedback_impact_score(forecast, offset)
    assert s["feasible_predicted_impact"]
    assert s["first_predicted_foot_substep"] == 29
    np.testing.assert_allclose(s["initial_window_robot_impulse_n_s"], [0.2, 0.04, 0])
    assert s["proxy_cost"] == pytest.approx(0.16)
    assert all(
        s[k] is False
        for k in (
            "final_m0_success_proven",
            "post80ms_stability_proven",
            "neural_policy_trained",
            "teacher_qualified",
            "hardware_authorized",
            "promotion_authorized",
        )
    )


@pytest.mark.parametrize("first", [None, 31, 39])
def test_no_or_incomplete_contact_is_not_feasible(first):
    forecast, offset = fixture(first=first)
    s = early_feedback_impact_score(forecast, offset)
    assert not s["feasible_predicted_impact"] and s["proxy_cost"] is None


@pytest.mark.parametrize("substep", [0, 10, 20, 30, 39])
@pytest.mark.parametrize("bad", ["nonfoot", "pelvis"])
def test_safety_covers_entire80ms_not_only_impact_window(substep, bad):
    f, offset = fixture()
    if bad == "nonfoot":
        f.arrays["nonfoot_robot_contact_over1n_after_substep"][substep] = True
    else:
        f.arrays["pelvis_height_solver_epoch_m"][substep] = 0.649
    s = early_feedback_impact_score(f, offset)
    assert not s["feasible_predicted_impact"] and s["proxy_cost"] is None


def test_action_penalty_accounts_for_three_periods_and_washout_is_enforced():
    offset = CAPS.copy()
    f, offset = fixture(offset=offset)
    s = early_feedback_impact_score(f, offset)
    assert s["action_regularizer"] == pytest.approx(29 * 3e-4)
    f.arrays["requested_offset_nm"][30, 0] = offset[0]
    with pytest.raises(ValueError, match="60ms-then-zero"):
        early_feedback_impact_score(f, offset)


@pytest.mark.parametrize("key", list(fixture()[0].arrays))
def test_rejects_short_or_wrong_typed_predictions(key):
    f, offset = fixture()
    f.arrays[key] = f.arrays[key][:-1]
    with pytest.raises(ValueError):
        early_feedback_impact_score(f, offset)


@pytest.mark.parametrize("bad", ["nan", "large_action", "changed_action", "negative_contact"])
def test_rejects_nonfinite_or_mismatched_action_and_contact(bad):
    f, offset = fixture()
    if bad == "nan":
        f.arrays["canonical_qvel_after_substep"][39, 35] = np.nan
    elif bad == "large_action":
        offset[0] = CAPS[0] + 0.01
    elif bad == "changed_action":
        f.arrays["requested_offset_nm"][0, 0] = 0.001
    else:
        f.arrays["foot_contact_peak_force_n_after_substep"][0, 1] = -1.0
    with pytest.raises(ValueError):
        early_feedback_impact_score(f, offset)
