"""Off-support phase choices must fall back to the original SONIC action."""

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.contextual_phase_policy_v2 import (
    FEATURE_NAMES,
    PHASE_ACTIONS,
    load_phase_actor_v2,
    support_aware_actions,
)
from rosclaw_soccer.sim.contracts import hash_json


def test_context_support_guard_returns_zero_outside_training_domain() -> None:
    weights = np.zeros((len(FEATURE_NAMES), len(PHASE_ACTIONS)))
    weights[0, 2] = 2.0
    training = np.zeros((35, len(FEATURE_NAMES)))
    training[:, 0] = 1
    query = np.asarray([[1, 0.1, 0, 0, 0], [1, 3, 0, 0, 0]])
    np.testing.assert_array_equal(support_aware_actions(query, weights, training, 0.5), [6.0, 0.0])
    with pytest.raises(ValueError):
        support_aware_actions(query, weights, training, float("inf"))


def test_v2_manifest_authentication_and_shape(tmp_path: Path) -> None:
    manifest = {
        "schema": "rsi_contextual_sonic_phase_actor_v2",
        "activation_ceiling": "SIM_ONLY",
        "feature_names": list(FEATURE_NAMES),
        "phase_actions_frames": list(PHASE_ACTIONS),
        "support_threshold": 0.5,
        "weights": np.zeros((len(FEATURE_NAMES), len(PHASE_ACTIONS))).tolist(),
        "support_contexts": np.zeros((35, len(FEATURE_NAMES))).tolist(),
        "promotion_authorized": False,
    }
    manifest["actor_hash"] = hash_json(manifest)
    path = tmp_path / "actor.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert load_phase_actor_v2(path)[0] == manifest["actor_hash"]
    manifest["support_threshold"] = 100
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unauthenticated"):
        load_phase_actor_v2(path)
