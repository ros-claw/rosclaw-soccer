"""Prospective collision timing must use only current ball/body state."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_time_phase_features import (
    ACTION_FEATURE_NAMES,
    current_context,
    fit_action_values,
    fit_contact_time,
    gait_phase_features,
    predict_contact_time,
    select_phase_actions,
)


def test_contact_time_and_gait_phase_features_are_bounded() -> None:
    n = 12
    root = np.zeros((n, 7))
    root_v = np.zeros((n, 6))
    root_v[:, 0] = 0.8
    ball = np.zeros((n, 3))
    ball[:, 0] = np.linspace(1.7, 2.5, n)
    ball_v = np.zeros((n, 3))
    ball_v[:, 0] = -0.5
    context = current_context(root, root_v, ball, ball_v)
    labels = np.linspace(42, 80, n)
    weights = fit_contact_time(context, labels)
    predicted = predict_contact_time(context, weights)
    assert np.isfinite(predicted).all()
    assert np.all((predicted >= 15) & (predicted <= 120))
    features = gait_phase_features(context, predicted)
    assert features.shape == (n, len(ACTION_FEATURE_NAMES))
    values = fit_action_values(features, np.ones((n, 3)), 1.0)
    assert set(select_phase_actions(features, values)).issubset({0.0, 3.0, 6.0})


def test_contact_time_rejects_future_or_invalid_observations() -> None:
    root = np.zeros((8, 7))
    velocity = np.zeros((8, 6))
    ball = np.zeros((8, 3))
    ball_velocity = np.zeros((8, 3))
    ball[0, 0] = float("nan")
    with pytest.raises(ValueError):
        current_context(root, velocity, ball, ball_velocity)
    ball[0, 0] = 2
    context = current_context(root, velocity, ball, ball_velocity)
    with pytest.raises(ValueError):
        fit_contact_time(context, np.full(8, 0))
    with pytest.raises(ValueError):
        predict_contact_time(context, np.zeros(5))
