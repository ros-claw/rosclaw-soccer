import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi import motor_bootstrap_network as base_module
from rosclaw_soccer.rsi.online_motor_actor_critic import (
    actor_parameters,
    make_model,
    terminal_return,
    update_from_physics,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_motor_bootstrap_network import bank


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("torch")
    base = base_module.fit_bootstrap(bank(), epochs=2)
    anchors = [[float(i) / 12] * 13 for i in range(9)]
    return make_model(base, anchors, "sha256:" + "a" * 64)


def outcome(reward=1.0):
    return dict(
        reward=reward,
        high_quality=True,
        clean_foot_only=True,
        minimum_pelvis_z_m=0.75,
        maximum_lateral_excursion_m=2.0,
    )


def test_physical_update_changes_plastic_context_but_not_nine_anchors(model):
    features = [0.9, -0.7] + [0.3] * 11
    before = actor_parameters(model, features)
    sample = dict(observation=features, normalized_action=[0.7] * 37, outcome=outcome())
    updated = update_from_physics(model, [sample], "sha256:" + "b" * 64)
    assert not np.array_equal(before, actor_parameters(updated, features))
    assert updated["generation"] == 1
    assert updated["parent_model_hash"] == model["model_hash"]
    for anchor in model["protected_features"]:
        assert np.array_equal(actor_parameters(model, anchor), actor_parameters(updated, anchor))
    assert updated["hardware_authorized"] is False
    assert updated["fresh_holdout_open_authorized"] is False


def test_repeated_identical_replay_is_not_extra_learning_or_conflicting_evidence(model):
    sample = dict(observation=[0.8] * 13, normalized_action=[0.1] * 37, outcome=outcome())
    once = update_from_physics(model, [sample], "sha256:" + "c" * 64)
    repeated = update_from_physics(model, [sample] * 4, "sha256:" + "c" * 64)
    assert once == repeated
    changed = copy.deepcopy(sample)
    changed["outcome"]["reward"] += 0.1
    with pytest.raises(ValueError, match="inconsistent"):
        update_from_physics(model, [sample, changed], "sha256:" + "c" * 64)


def test_weight_or_encoder_tampering_cannot_reuse_old_protection(model):
    changed = copy.deepcopy(model)
    changed["base_model"]["actor"]["layers"][0]["weight"][0][0] += 0.1
    changed["base_model"]["model_hash"] = hash_json(
        {k: v for k, v in changed["base_model"].items() if k != "model_hash"}
    )
    changed["model_hash"] = hash_json({k: v for k, v in changed.items() if k != "model_hash"})
    with pytest.raises(ValueError, match="encoder"):
        validate_model(changed)


@pytest.mark.parametrize("field", ["runtime_execution_authorized", "promotion_authorized"])
def test_resealed_model_cannot_grant_authority(model, field):
    changed = copy.deepcopy(model)
    changed[field] = True
    changed["model_hash"] = hash_json({k: v for k, v in changed.items() if k != "model_hash"})
    with pytest.raises(ValueError):
        validate_model(changed)


def test_terminal_reward_is_physical_and_rejects_nonfinite_values():
    assert terminal_return(outcome()) == 11
    unsafe = outcome()
    unsafe.update(clean_foot_only=False, minimum_pelvis_z_m=0.5, maximum_lateral_excursion_m=5)
    assert terminal_return(unsafe) == -117
    with pytest.raises(ValueError):
        terminal_return(outcome(float("nan")))
