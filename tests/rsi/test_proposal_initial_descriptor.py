"""Commitment equivalence, not a validation bypass or physical certificate."""

import copy

import pytest

from rosclaw_soccer.rsi.proposal_memory_motor import (
    _initial_descriptor,
    initial_model,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.mark.parametrize("profile", ["uniform-frame", "equal-contact-phase-mass"])
def test_internal_descriptor_exactly_matches_public_owned_builder(current, profile):  # noqa: F811
    actor = current[0]
    descriptor = _initial_descriptor(actor, maximum_mean_kl=0.05, loss_weighting_profile=profile)
    built = initial_model(actor, maximum_mean_kl=0.05, loss_weighting_profile=profile)
    assert built["model_hash"] == hash_json(descriptor)
    assert {k: v for k, v in built.items() if k != "model_hash"} == descriptor
    assert built["initial_actor"] is not actor
    assert built["residual_layers"] is not actor["residual_layers"]
    assert built["residual_layers"] is not built["initial_actor"]["residual_layers"]
    frozen_parent_layers = copy.deepcopy(built["initial_actor"]["residual_layers"])
    built["residual_layers"][0]["weight"][0][0] += 0.1
    assert descriptor["residual_layers"] == actor["residual_layers"]
    assert built["initial_actor"]["residual_layers"] == frozen_parent_layers


def test_public_builder_still_rejects_invalid_actor_and_resealed_initial_model(current):  # noqa: F811
    actor = copy.deepcopy(current[0])
    actor["hardware_authorized"] = True
    actor.pop("model_hash")
    actor["model_hash"] = hash_json(actor)
    with pytest.raises(ValueError):
        initial_model(actor, maximum_mean_kl=0.05)
    built = initial_model(current[0], maximum_mean_kl=0.05)
    built["initial_actor"] = actor
    built.pop("model_hash")
    built["model_hash"] = hash_json(built)
    with pytest.raises(ValueError):
        validate_model(built)
