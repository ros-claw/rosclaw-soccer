"""Sealed proprioceptive option model is finite, bounded and tamper-evident."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.precontact_proprio_policy import (
    FEATURE_NAMES,
    V300_HASH,
    proprio_vector,
    risk_probability,
    validate_policy,
)
from rosclaw_soccer.sim.contracts import hash_json


def _policy() -> dict:
    policy = {
        "schema": "rsi_precontact_proprio_approach_policy_v1",
        "activation_ceiling": "SIM_ONLY",
        "training_report_hash": V300_HASH,
        "feature_names": list(FEATURE_NAMES),
        "decision_frame": 30,
        "threshold": 0.2,
        "aggressive_gain": 1.2,
        "fallback_gain": 0.8,
        "mean": [0.0] * 13,
        "scale": [1.0] * 13,
        "coefficients": [0.0] * 13,
        "intercept": 0.0,
        "promotion_authorized": False,
    }
    policy["policy_hash"] = hash_json(policy)
    return policy


def test_policy_hash_and_probability() -> None:
    policy = _policy()
    assert validate_policy(policy)[1] == policy["policy_hash"]
    assert risk_probability(policy, (0.0,) * 13) == 0.5
    policy["threshold"] = 0.5
    with pytest.raises(ValueError, match="unsealed"):
        validate_policy(policy)


def test_proprio_vector_uses_precontact_geometry() -> None:
    root = np.zeros(7)
    velocity = np.zeros(6)
    ball20 = np.asarray((2.0, -0.1, 0.11))
    ball30 = np.asarray((1.8, -0.1, 0.11))
    ball_velocity = np.asarray((-0.6, 0.0, 0.0))
    geometry = np.zeros((4, 3))
    geometry[:, 2] = 0.2
    features = proprio_vector(root, velocity, ball20, ball30, ball_velocity, geometry, geometry)
    assert len(features) == 13
    assert features[:3] == (1.8, -0.1, -0.6)
    assert features[-1] > 0
