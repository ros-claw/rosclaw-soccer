import numpy as np
import pytest
from test_shared_native_motor_world import world

from rosclaw_soccer.rsi.shared_target_forecast import SharedTargetForecaster, compiled_model_hash


def test_forecast_does_not_modify_donor_and_matches_actual_pd_execution_bitwise():
    w = world()
    w.advance({p: np.zeros(29) for p in w.layouts})
    before = w.data.qpos.copy(), w.data.qvel.copy(), w.data.ctrl.copy(), w.data.time
    predictor = SharedTargetForecaster(
        w, expected_model_hash=compiled_model_hash(w.model), native_step_budget=20
    )
    targets = {p: np.linspace(-0.02, 0.02, 29) for p in w.layouts}
    predicted = predictor.forecast(targets)
    for expected, actual in zip(before[:3], (w.data.qpos, w.data.qvel, w.data.ctrl), strict=True):
        np.testing.assert_array_equal(expected, actual)
    assert w.data.time == before[3] and w.native_steps == 10
    actual = w.advance(targets)
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        np.testing.assert_array_equal(predicted[key], actual[key])
    assert predicted["donor_control_frame"] == 1
    assert predicted["predicted_time"] == w.data.time
    assert predicted["prediction_is_not_actual_execution"]
    assert predictor.counters()["integrated_native_steps"] == 10


def test_forecasts_are_independent_copies_and_budget_is_enforced():
    w = world()
    predictor = SharedTargetForecaster(
        w, expected_model_hash=compiled_model_hash(w.model), native_step_budget=20
    )
    targets = {p: np.zeros(29) for p in w.layouts}
    first = predictor.forecast(targets)
    second = predictor.forecast(targets)
    np.testing.assert_array_equal(first["qpos"], second["qpos"])
    assert w.native_steps == 0
    with pytest.raises(RuntimeError, match="exhausted"):
        predictor.forecast(targets)
    assert predictor.counters()["attempted_native_steps"] == 20


def test_invalid_target_is_charged_without_hidden_physics_or_mutating_donor():
    w = world()
    predictor = SharedTargetForecaster(
        w, expected_model_hash=compiled_model_hash(w.model), native_step_budget=10
    )
    with pytest.raises(ValueError):
        predictor.forecast({"a/": np.zeros(29)})
    assert predictor.counters()["attempted_native_steps"] == 10
    assert predictor.counters()["integrated_native_steps"] == 0
    assert w.native_steps == 0 and w.data.time == 0


def test_changed_model_is_rejected_before_forecast_allocation():
    w = world()
    predictor = SharedTargetForecaster(
        w, expected_model_hash=compiled_model_hash(w.model), native_step_budget=10
    )
    w.model.body_mass[w.layouts["a/"].pelvis] *= 1.1
    with pytest.raises(ValueError, match="changed"):
        predictor.forecast({p: np.zeros(29) for p in w.layouts})
    assert predictor.counters()["attempted_native_steps"] == 0
