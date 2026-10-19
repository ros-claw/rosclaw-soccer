"""Structural integration tests only, not G1 task/teacher successes."""

import inspect

import mujoco
import numpy as np
import pytest

from rosclaw_soccer.physics.g1_contact_continuation import G1ContactContinuationForecaster
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS
from tests.physics import test_g1_pd_torque_forecast as structural


@pytest.fixture
def structural_model():
    return structural.structural_model.__wrapped__()


def forecaster(structural_model, budget=40):
    model, h = structural_model
    return G1ContactContinuationForecaster(
        model,
        expected_world_hash=h,
        kp=np.full(29, 10.0),
        kd=np.full(29, 0.2),
        max_forecast_steps=budget,
    )


@pytest.mark.parametrize("offset_value", [0.0, 1.0])
def test_two_chunks_equal_uninterrupted_twenty_substeps(structural_model, offset_value):
    model, _ = structural_model
    donor = mujoco.MjData(model)
    donor.qvel[6:35] = np.linspace(-0.1, 0.1, 29)
    mujoco.mj_forward(model, donor)
    actual = mujoco.MjData(model)
    mujoco.mj_copyData(actual, model, donor)
    target, offset = np.full(29, 0.1), np.full(29, offset_value)
    f = forecaster(structural_model)
    prediction = f.forecast(donor, target, offset)
    arrays = prediction.substep_arrays()
    for step in range(20):
        raw = (target - actual.qpos[7:36]) * 10.0 - actual.qvel[6:35] * 0.2
        tau = np.clip(
            raw + offset if np.any(offset) else raw,
            -np.asarray(G1_HARD_TORQUE_LIMITS),
            G1_HARD_TORQUE_LIMITS,
        )
        actual.ctrl[:] = tau
        mujoco.mj_step(model, actual)
        assert actual.qpos.tobytes() == arrays["canonical_qpos_after_substep"][step].tobytes()
        assert actual.qvel.tobytes() == arrays["canonical_qvel_after_substep"][step].tobytes()
    end = np.empty_like(prediction.second.end_integration_state)
    mujoco.mj_getState(model, actual, end, mujoco.mjtState.mjSTATE_INTEGRATION)
    assert end.tobytes() == prediction.second.end_integration_state.tobytes()
    assert donor.time == 0 and f.attempted_steps == f.integrated_steps == 20
    again = f.forecast(donor, target, offset)
    assert again.second.end_integration_state.tobytes() == end.tobytes()
    assert f.forecast_attempts == 2
    with pytest.raises(RuntimeError, match="budget"):
        f.forecast(donor, target, offset)
    assert f.attempted_steps == 40


@pytest.mark.parametrize("budget", [0, 10, 30, True, 20.0, 600020])
def test_whole_bounded_forecast_budget(structural_model, budget):
    with pytest.raises(ValueError, match="budget"):
        forecaster(structural_model, budget)


def test_no_future_trace_or_targets_in_prediction_api():
    assert set(inspect.signature(G1ContactContinuationForecaster.forecast).parameters) == {
        "self",
        "data",
        "current_target_rad",
        "offset_nm",
    }
