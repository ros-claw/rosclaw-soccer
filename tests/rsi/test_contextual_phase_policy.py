"""Causal, bounded snapshot-context phase actor tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.contextual_phase_policy import (
    FEATURE_NAMES,
    PHASE_ACTIONS,
    context_features,
    fit_action_values,
    load_phase_actor,
    select_actions,
)
from rosclaw_soccer.sim.contracts import hash_json


def test_context_uses_current_body_and_ball_only() -> None:
    root = np.zeros((2, 7))
    ball = np.asarray([[2.0, 0.1, 0.11], [2.5, -0.1, 0.11]])
    velocity = np.asarray([[-0.5, 0, 0], [-0.3, 0, 0]])
    feet = np.zeros((2, 4, 3))
    feet[:, 1, 0] = -0.1
    features = context_features(root, ball, velocity, feet)
    assert features.shape == (2, len(FEATURE_NAMES))
    np.testing.assert_allclose(features[:, 1], [0, 1])
    np.testing.assert_allclose(features[:, 2], [0.5, -0.5])
    with pytest.raises(ValueError):
        context_features(root, ball[:, :2], velocity, feet)


def test_ridge_actor_selects_measured_action_and_rejects_invalid() -> None:
    features = np.asarray([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 2, 0, 0, 0]])
    rewards = np.asarray([[1, 0, 0], [0, 0, 1], [0, 0, 2]])
    weights = fit_action_values(features, rewards, 1.0)
    selected = select_actions(features, weights)
    assert set(selected).issubset(set(PHASE_ACTIONS))
    with pytest.raises(ValueError):
        fit_action_values(features, rewards, 0.0)


def test_context_actor_manifest_hash_fails_closed(tmp_path: Path) -> None:
    manifest = {
        "schema": "rsi_contextual_sonic_phase_actor_v1",
        "activation_ceiling": "SIM_ONLY",
        "phase_actions_frames": list(PHASE_ACTIONS),
        "feature_names": list(FEATURE_NAMES),
        "weights": np.zeros((len(FEATURE_NAMES), len(PHASE_ACTIONS))).tolist(),
        "promotion_authorized": False,
    }
    manifest["actor_hash"] = hash_json(manifest)
    path = tmp_path / "actor.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_phase_actor(path)[0] == manifest["actor_hash"]
    manifest["weights"][0][0] = 4
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unauthenticated"):
        load_phase_actor(path)
