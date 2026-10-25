import numpy as np
import pytest

import rosclaw_soccer.rsi.sequential_joint_response as module


def observed(f=0.3, lateral=0.8, force=50.0, *, complete=True, healthy=True):
    prediction = dict(
        controls=128,
        donor_control_frame=80,
        ball_forces=np.full((1280, 1, 3), force, dtype=np.float64),
    )
    window = dict(
        predicted_body_healthy=healthy,
        quality_observed=complete,
        forward_60_m=f,
        lateral_60_m=lateral,
        window_pass=bool(
            healthy and complete and f >= 1 and abs(lateral) <= 0.3 * f and force <= 1
        ),
    )
    return prediction, window


def test_four_round_limit_accepts_only_fresh_forecast_improvement_and_fixed_scale():
    calls = []

    def evaluate(name, coeff):
        assert coeff.shape == (2, 29) and not coeff[1].any()
        assert abs(coeff).max() <= 0.06
        calls.append((name, coeff.copy()))
        total = float(coeff[0].sum())
        return observed(0.3 + 3 * total, 0.8 - 2 * total, 50 - 50 * total)

    result = module.search_sequential_joint_response(*observed(), evaluate)
    assert len(result["rounds"]) == 4 and len(calls) <= 236
    assert result["fixed_initial_force_scale"] == 50
    assert result["status"] == "FOUR_ROUND_LIMIT_REQUIRES_ACTUAL_EXAM"
    previous = result["rounds"][0]["observed_center_violation"]
    for row in result["rounds"]:
        assert row["accepted_new_center"]
        assert row["accepted_observed_violation"] < previous
        assert row["fit"]["force_scale"] == 50
        previous = row["accepted_observed_violation"]
    assert not result["actual_execution_success_proven"]


def test_optimistic_linear_model_cannot_replace_failed_physical_feedback(monkeypatch):
    monkeypatch.setattr(
        module,
        "restore_joint_feasibility",
        lambda *a, **k: (np.full(29, 0.001), {"predicted_violation": 0.0}),
    )
    result = module.search_sequential_joint_response(*observed(), lambda name, coeff: observed())
    assert result["newly_requested_forecasts"] == 59
    assert result["status"] == "ABSTAIN_NO_NEW_NN_PHYSICAL_PROGRESS"
    assert not result["rounds"][0]["accepted_new_center"]


@pytest.mark.parametrize("reason", ["censored", "body_failed"])
def test_invalid_forecast_never_becomes_new_center(reason):
    def evaluate(name, coeff):
        return observed(100, 0, 0, complete=reason != "censored", healthy=reason != "body_failed")

    result = module.search_sequential_joint_response(*observed(), evaluate)
    assert result["newly_requested_forecasts"] == 58
    assert not result["rounds"][0]["accepted_new_center"]


def test_predicted_success_is_not_claimed_as_actual_task_success():
    result = module.search_sequential_joint_response(*observed(), lambda n, c: observed(2, 0, 0))
    assert result["newly_requested_forecasts"] == 58
    assert len(result["rounds"]) == 1
    assert result["status"] == "PREDICTED_TASK_FEASIBLE_REQUIRES_ACTUAL_EXAM"
    assert not result["actual_execution_success_proven"]


def test_censored_parent_does_not_fabricate_a_center():
    def unexpected(*args):
        raise AssertionError("no meaningful complete center")

    result = module.search_sequential_joint_response(*observed(complete=False), unexpected)
    assert result["newly_requested_forecasts"] == 0 and not result["rounds"]


def test_independent_readback_rejects_changed_coefficients_and_fake_progress():
    import copy

    from scripts.rsi_review_joint_response_pilot import verify_sequential_response

    prediction, window = observed()
    windows = [dict(name="parent", coefficients=None, outcome=window) for _ in range(37)]
    forecasts = [prediction for _ in range(37)]

    def evaluate(name, coeff):
        total = float(coeff[0].sum())
        pred, win = observed(0.3 + 3 * total, 0.8 - 2 * total, 50 - 50 * total)
        windows.append(dict(name=name, coefficients=coeff.tolist(), outcome=win))
        forecasts.append(pred)
        return pred, win

    report = module.search_sequential_joint_response(prediction, window, evaluate)
    verify_sequential_response(windows, forecasts, report)
    bad = copy.deepcopy(report)
    bad["rounds"][0]["accepted_observed_violation"] = 0
    with pytest.raises(ValueError, match="report differs"):
        verify_sequential_response(windows, forecasts, bad)
    changed = copy.deepcopy(windows)
    changed[37]["coefficients"][0][0] += 0.001
    with pytest.raises(ValueError, match="coefficients differ"):
        verify_sequential_response(changed, forecasts, report)
