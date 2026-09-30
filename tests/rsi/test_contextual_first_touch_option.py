"""First-touch option learning sees only measured precontact features and outcomes."""

from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.rsi.contextual_first_touch_option import (
    first_touch_reward,
    fit_contextual_option,
)


def test_reward_keeps_safety_and_contact_quality_ahead_of_distance() -> None:
    arm = {
        "minimum_pelvis_z_m": 0.69,
        "contact_body_indices": [1],
        "first_contact_frame": 85,
        "forward_60_m": 1.0,
        "lateral_60_m": 0.0,
        "max_lateral_excursion_m": 1.0,
    }
    assert first_touch_reward(arm) == pytest.approx(3.0)
    arm["max_lateral_excursion_m"] = 4.01
    assert first_touch_reward(arm) == -4.0
    arm["max_lateral_excursion_m"] = 1.0
    arm["contact_body_indices"] = [5]
    assert first_touch_reward(arm) == -2.0
    arm["first_contact_frame"] = None
    arm["contact_body_indices"] = []
    assert first_touch_reward(arm) == -3.0
    arm["minimum_pelvis_z_m"] = 0.64
    assert first_touch_reward(arm) == -10.0
    arm["minimum_pelvis_z_m"] = float("nan")
    with pytest.raises(ValueError):
        first_touch_reward(arm)


def test_regularized_option_uses_training_features_only_and_parent_ties() -> None:
    features = np.zeros((4, 9))
    features[:, 1] = [0, 0.1, 5, 5.1]
    rewards = np.array([[0, 3, -2], [0, 2, -2], [1, -2, 3], [1, -2, 2]])
    model = fit_contextual_option(features, rewards)
    assert model.choose(features[0]) == 1
    assert model.choose(features[3]) == 2
    assert fit_contextual_option(features, np.zeros((4, 3))).choose(features[0]) == 0
    assert fit_contextual_option(features, rewards[:, :2], action_count=2).choose(features[0]) == 1
    with pytest.raises(ValueError):
        model.choose(np.full(9, np.nan))
    with pytest.raises(ValueError):
        fit_contextual_option(features, rewards[:3])
