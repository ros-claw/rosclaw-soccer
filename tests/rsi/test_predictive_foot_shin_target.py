import numpy as np
import pytest

from rosclaw_soccer.rsi.predictive_foot_shin_target import predictive_foot_shin_target


def inputs():
    foot = np.zeros((3, 29), dtype=np.float64)
    foot[:, :3] = np.eye(3)
    shin = np.zeros((2, 29), dtype=np.float64)
    shin[0, 3] = 1
    return dict(
        nominal=np.zeros(29),
        previous_delta=np.zeros(29),
        joint_ranges=np.tile([-2.0, 2.0], (29, 1)),
        foot_jacobian=foot,
        foot_displacement=np.array([0.02, 0.0, 0.0]),
        shin_gap=np.array([0.1, 0.3]),
        shin_gap_rate=np.array([-1.0, 0.0]),
        shin_gap_jacobian=shin,
        foot=0,
        enabled=True,
    )


def test_closing_shin_gets_positive_clearance_and_leg_only_bounded_target():
    out = predictive_foot_shin_target(**inputs())
    assert out["predicted_shin_gap"][0] == pytest.approx(-0.02)
    assert out["desired_shin_clearance"][0] == pytest.approx(0.07)
    assert out["applied_delta"][3] > 0
    assert np.max(np.abs(out["applied_delta"])) <= 0.025 + 1e-12
    assert np.all(out["applied_delta"][6:] == 0)
    assert not out["fallback"]


def test_no_closing_risk_requires_no_shin_clearance():
    args = inputs()
    args["shin_gap_rate"][:] = 0
    out = predictive_foot_shin_target(**args)
    assert np.all(out["desired_shin_clearance"] == 0)
    assert out["applied_delta"][3] == 0


def test_disable_slews_previous_correction_toward_zero():
    args = inputs()
    args["enabled"] = False
    args["previous_delta"][3] = 0.05
    assert predictive_foot_shin_target(**args)["applied_delta"][3] == pytest.approx(0.025)


def test_outside_parent_axis_is_frozen_without_disabling_other_axes():
    args = inputs()
    args["nominal"][0] = 3
    out = predictive_foot_shin_target(**args)
    assert not out["fallback"]
    assert out["frozen_parent_axes"][0]
    assert out["target"][0] == args["nominal"][0]
    assert out["applied_delta"][3] > 0


def test_nonzero_prior_correction_with_infeasible_bounds_falls_back():
    args = inputs()
    args["nominal"][0] = 3
    args["previous_delta"][0] = 0.01
    out = predictive_foot_shin_target(**args)
    assert out["fallback"]
    np.testing.assert_array_equal(out["target"], args["nominal"])


@pytest.mark.parametrize("fault", ["nan", "wrong_leg", "unbounded_foot"])
def test_bad_inputs_fail_closed(fault):
    args = inputs()
    if fault == "nan":
        args["shin_gap_rate"][0] = np.nan
    elif fault == "wrong_leg":
        args["previous_delta"][6] = 0.01
    else:
        args["foot_displacement"][0] = 0.081
    with pytest.raises(ValueError):
        predictive_foot_shin_target(**args)
