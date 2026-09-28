"""Causal, safety-gated physical action-library learner contracts."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.rsi_team_contextual_action_learner import causal_features, choose_arm


def test_features_use_measured_entry_feet_and_ball_only() -> None:
    state = [0.0] * 24
    state[6:9] = [2.0, -0.4, 0.11]
    state[9] = -0.5
    state[12:18] = [1.0, 0.0, 0.2, 1.2, -0.3, 0.1]
    state[18:24] = [0.3, 0.2, 0.0, 0.1, 0.0, 0.0]
    features = causal_features(state)
    assert features.shape == (8,)
    assert np.allclose(features[:4], (1.0, -0.4, 0.1, -0.5))
    assert np.allclose(features[-2:], (0.3, 0.2))
    with pytest.raises(ValueError):
        causal_features(state[:-1])


def test_selector_rejects_unsafe_neighbors_even_if_average_reward_is_positive() -> None:
    features = np.zeros((4, 8))
    features[:, 0] = (0.1, 0.2, 0.3, 0.4)
    outcomes = {
        "fast": [
            {"safe": True, "useful_pass": True, "foot_contact_frames": [1]},
            {"safe": True, "useful_pass": True, "foot_contact_frames": [1]},
            {"safe": False, "useful_pass": False, "foot_contact_frames": []},
        ],
        "safe": [
            {"safe": True, "useful_pass": False, "foot_contact_frames": [1]} for _ in range(3)
        ],
    }
    selected = choose_arm(features[3], features, outcomes, [0, 1, 2], ["fast", "safe"])
    assert selected["arm"] == "safe"
    assert selected["abstained"] is False


def test_selector_abstains_when_no_safe_positive_arm() -> None:
    features = np.zeros((4, 8))
    outcomes = {
        "unsafe": [
            {"safe": False, "useful_pass": False, "foot_contact_frames": []} for _ in range(3)
        ]
    }
    selected = choose_arm(features[3], features, outcomes, [0, 1, 2], ["unsafe"])
    assert selected["arm"] is None
    assert selected["abstained"] is True
