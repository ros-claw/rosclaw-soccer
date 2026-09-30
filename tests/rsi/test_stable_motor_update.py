import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi import motor_bootstrap_network as base_module
from rosclaw_soccer.rsi.online_motor_actor_critic import actor_parameters, make_model
from rosclaw_soccer.rsi.stable_motor_update import stable_update, validate_stable_model
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_motor_bootstrap_network import bank


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("torch")
    base = base_module.fit_bootstrap(bank(), epochs=2)
    return make_model(base, [[i / 12] * 13 for i in range(9)], "sha256:" + "a" * 64)


def sample(action, quality, clean, reward):
    return dict(
        observation=[0.9, -0.7] + [0.3] * 11,
        normalized_action=[action] * 37,
        outcome=dict(
            reward=reward,
            high_quality=quality,
            clean_foot_only=clean,
            minimum_pelvis_z_m=0.75,
            maximum_lateral_excursion_m=2.0,
        ),
    )


def test_stale_baseline_cannot_make_low_quality_and_high_quality_equal(model):
    failed = [sample(-0.2 + i * 0.005, False, False, -2) for i in range(32)]
    replay = failed + [sample(-0.7, False, True, 3), sample(0.7, True, True, 3)]
    trained = stable_update(model, replay, "sha256:" + "b" * 64)
    actual = actor_parameters(trained, replay[0]["observation"])
    expected = np.asarray([0.7 * 0.16] * 36 + [0.7 * 0.3 - 0.05])
    assert np.allclose(actual, expected, atol=1e-6, rtol=0)
    for anchor in model["protected_features"]:
        assert np.array_equal(actor_parameters(model, anchor), actor_parameters(trained, anchor))
    assert trained["runtime_execution_authorized"] is False


def test_learning_rule_source_cannot_be_self_resealed(model):
    trained = stable_update(model, [sample(0.4, True, True, 3)], "sha256:" + "b" * 64)
    changed = copy.deepcopy(trained)
    changed["weighting_source_hash"] = "sha256:" + "c" * 64
    changed["model_hash"] = hash_json({k: v for k, v in changed.items() if k != "model_hash"})
    with pytest.raises(ValueError):
        validate_stable_model(changed)


def test_inconsistent_replay_is_not_silently_deduplicated(model):
    before = sample(0.4, True, True, 3)
    after = sample(0.4, False, True, 3)
    with pytest.raises(ValueError, match="conflicting"):
        stable_update(model, [before, after], "sha256:" + "b" * 64)
