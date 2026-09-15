"""Numerical feature contracts, not physical qualification or model inference."""

from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_runup import (
    _VARIANTS,
    G1SonicRunupConfig,
    G1SonicRunupController,
)


@pytest.fixture(params=["low_latency", "sonic_v1_1"])
def feature_controller(request):
    pytest.importorskip("mujoco")
    # No weights, planner, transport, runtime or simulated robot are opened.
    motor = object.__new__(G1SonicRunupController)
    motor.config = G1SonicRunupConfig(model_variant=request.param)
    motor._variant = _VARIANTS[request.param]
    motor.default_angles = np.zeros(29)
    motor.reference = np.zeros((100, 36))
    motor.reference[:, 3] = 1.0
    motor.reference[:, 7:] = np.arange(100)[:, None] * 0.001
    return motor


def measured_state():
    qpos = np.zeros(43)
    qpos[3] = 1.0
    return SimpleNamespace(qpos=qpos, qvel=np.zeros(41))


def test_g1_features_do_not_observe_absolute_planar_tracking_error(feature_controller):
    motor = feature_controller
    state = measured_state()
    action = np.zeros(29)
    encoded = motor._encoder_observation(state, 0)
    history = motor._history_entry(state, action)

    state.qpos[:2] = [3.0, -2.0]
    state.qvel[:3] = [0.8, -0.6, 0.1]
    np.testing.assert_array_equal(encoded, motor._encoder_observation(state, 0))
    for before, after in zip(history, motor._history_entry(state, action), strict=True):
        np.testing.assert_array_equal(before, after)

    motor.reference[:, :2] += [5.0, 7.0]
    np.testing.assert_array_equal(encoded, motor._encoder_observation(state, 0))
    # Translating references alone cannot repair this body's planar error.
    # An outer planner must regenerate joint/orientation motion from feedback.


def test_future_joint_motion_is_visible_before_its_execution_frame(feature_controller):
    motor = feature_controller
    state = measured_state()
    before = motor._encoder_observation(state, 0)
    future_frame = 9 * motor._variant.reference_stride
    motor.reference[future_frame, 7] += 0.1
    assert not np.array_equal(before, motor._encoder_observation(state, 0))
    # A future splice inside look-ahead can affect the current action; the
    # splice's execution time is not a guaranteed reaction delay.


def test_joint_motion_and_angular_velocity_remain_observable(feature_controller):
    motor = feature_controller
    state = measured_state()
    before = motor._history_entry(state, np.zeros(29))
    state.qvel[3] = 0.2
    state.qpos[7] = 0.1
    after = motor._history_entry(state, np.zeros(29))
    assert not np.array_equal(before[0], after[0])
    assert not np.array_equal(before[1], after[1])
