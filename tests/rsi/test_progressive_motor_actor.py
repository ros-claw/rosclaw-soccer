import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.motor_bootstrap_network import fit_bootstrap
from rosclaw_soccer.rsi.online_motor_actor_critic import actor_parameters as parent_parameters
from rosclaw_soccer.rsi.online_motor_actor_critic import make_model as parent_model
from rosclaw_soccer.rsi.progressive_motor_actor import (
    DIMENSION,
    actor_parameters,
    make_model,
    update_from_physics,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_motor_bootstrap_network import bank
from tests.rsi.test_online_motor_actor_critic import outcome


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("torch")
    base = fit_bootstrap(bank(), epochs=2)
    rng = np.random.default_rng(312)
    contexts = rng.normal(size=(52, 13)).tolist()
    parent = parent_model(base, contexts[:9], "sha256:" + "a" * 64)
    return make_model(
        parent,
        contexts,
        contexts[:38],
        learning_report_hash="sha256:" + "b" * 64,
        fresh_protocol_hash="sha256:" + "c" * 64,
    )


def test_expanded_capacity_preserves_parent_before_learning(model):
    plane = validate_model(model)
    assert plane.dimension == DIMENSION
    assert DIMENSION - plane.rank >= 40
    for observation in model["protected_features"] + [[0.5] * 13]:
        assert np.array_equal(
            actor_parameters(model, observation),
            parent_parameters(model["frozen_parent"], observation),
        )


def test_physical_update_learns_without_changing_38_anchors(model):
    observation = [0.7, -0.4] + [0.2] * 11
    sample = dict(observation=observation, normalized_action=[0.6] * 37, outcome=outcome())
    updated = update_from_physics(model, [sample], "sha256:" + "d" * 64)
    assert not np.array_equal(
        actor_parameters(model, observation), actor_parameters(updated, observation)
    )
    for anchor in model["protected_features"]:
        assert np.array_equal(actor_parameters(model, anchor), actor_parameters(updated, anchor))
    assert updated["frozen_parent"] == model["frozen_parent"]
    assert updated["generation"] == 1
    assert updated["fresh_holdout_open_authorized"] is False


def test_repeated_samples_do_not_become_extra_learning(model):
    sample = dict(observation=[0.2] * 13, normalized_action=[0.5] * 37, outcome=outcome())
    assert update_from_physics(model, [sample], "sha256:" + "e" * 64) == update_from_physics(
        model, [sample] * 5, "sha256:" + "e" * 64
    )
    changed = copy.deepcopy(sample)
    changed["outcome"]["reward"] += 1
    with pytest.raises(ValueError, match="inconsistent"):
        update_from_physics(model, [sample, changed], "sha256:" + "e" * 64)


@pytest.mark.parametrize(
    "field", ["promotion_authorized", "hardware_authorized", "fresh_holdout_open_authorized"]
)
def test_resealing_cannot_grant_authority(model, field):
    changed = copy.deepcopy(model)
    changed[field] = True
    changed["model_hash"] = hash_json({k: v for k, v in changed.items() if k != "model_hash"})
    with pytest.raises(ValueError):
        validate_model(changed)


def test_resealed_encoder_cannot_reuse_old_protection(model):
    changed = copy.deepcopy(model)
    changed["encoder"]["weight"][0][0] += 0.1
    changed["model_hash"] = hash_json({k: v for k, v in changed.items() if k != "model_hash"})
    with pytest.raises(ValueError, match="encoder"):
        validate_model(changed)


def test_invalid_action_and_observation_fail_closed(model):
    with pytest.raises(ValueError):
        actor_parameters(model, [float("nan")] * 13)
    with pytest.raises(ValueError):
        update_from_physics(
            model,
            [dict(observation=[0.2] * 13, normalized_action=[2.0] * 37, outcome=outcome())],
            "sha256:" + "f" * 64,
        )
