import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.extended_proposal_motor import (
    CompiledExtendedProposalMotor,
    make_model,
    make_preview,
    validate_model,
)
from rosclaw_soccer.rsi.proposal_decoder_selection import select_proposal_decoder
from rosclaw_soccer.rsi.proposal_memory_motor import initial_model
from rosclaw_soccer.rsi.proposal_memory_motor import make_preview as parent_preview
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_correlated_gradient_numerical_parity import conditional_fixture_batch
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def parent(request):
    return initial_model(request.getfixturevalue("current")[0], maximum_mean_kl=0.05)


def test_initial_extended_copy_has_identical_raw_law_and_independent_history(request):
    old = parent(request)
    value = make_model(old, maximum_inner_steps=1600)
    before = copy.deepcopy(value)
    original = select_proposal_decoder(parent_preview(old), implementation="bounded_snapshot")
    a, b = (CompiledExtendedProposalMotor(make_preview(value)) for _ in range(2))
    for phase in range(3):
        for query in np.random.default_rng(650).normal(size=(6, 134)):
            assert np.array_equal(a.raw_mean(query, phase), original.raw_mean(query, phase))
    assert a._memory is not b._memory
    assert a._parent._memory is not b._parent._memory
    assert a._guard.to_dict() == original._guard.to_dict()
    assert value == before
    assert value["frozen_parent"] == old
    assert (
        make_preview(value)["step_motor_proof"]["qualification"]
        == "UNQUALIFIED_SIM_EXTENDED_OPTIMIZER"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("hardware_authorized", True),
        ("promotion_authorized", True),
        ("runtime_execution_authorized", True),
        ("maximum_inner_steps", True),
        ("maximum_inner_steps", 2561),
        ("physical_action_bounds_changed", True),
        ("protected_memory_changed", True),
    ],
)
def test_resealed_capacity_and_authority_changes_rejected(request, field, value):
    artifact = make_model(parent(request), maximum_inner_steps=1600)
    artifact[field] = value
    artifact.pop("model_hash")
    artifact["model_hash"] = hash_json(artifact)
    with pytest.raises(ValueError):
        validate_model(artifact)


def test_unreceipted_or_malformed_plastic_changes_rejected(request):
    original = parent(request)
    changed = copy.deepcopy(original["residual_layers"])
    changed[-1]["bias"][0] = 0.1
    with pytest.raises(ValueError, match="unreceipted"):
        make_model(original, maximum_inner_steps=1600, residual_layers=changed)
    with pytest.raises(ValueError, match="receipt"):
        make_model(original, maximum_inner_steps=1600, learning_receipt={"algorithm": "fake"})


def test_actual_extended_fit_and_resealed_receipt_corruption(request):
    pytest.importorskip("torch")
    from rosclaw_soccer.rsi.extended_proposal_learning import fit_extended_update

    original = parent(request)
    data = conditional_fixture_batch(request.getfixturevalue("smooth_parent"))
    learned = fit_extended_update(
        original,
        data,
        batch_hash="sha256:" + "b" * 64,
        behavior_model_hash=original["model_hash"],
        critic_evidence_hash="sha256:" + "c" * 64,
        maximum_inner_steps=192,
    )
    assert learned["frozen_parent"] == original
    assert learned["residual_layers"] != original["residual_layers"]
    assert learned["learning_receipt"]["requested_inner_optimizer_steps"] == 192
    for key, value in (
        ("kl_budget_reset_between_passes", True),
        ("behavior_model_hash", "sha256:" + "d" * 64),
        ("exact_mean_marginal_kl", 0.051),
        ("physical_batch_verified", True),
        ("future_event_is_actor_observation", True),
        ("completed_optimizer_steps", 193),
    ):
        forged = copy.deepcopy(learned)
        forged["learning_receipt"][key] = value
        forged.pop("model_hash")
        forged["model_hash"] = hash_json(forged)
        with pytest.raises(ValueError):
            validate_model(forged)
    with pytest.raises(ValueError, match="immediate"):
        fit_extended_update(
            original,
            data,
            batch_hash="sha256:" + "b" * 64,
            behavior_model_hash="sha256:" + "d" * 64,
            critic_evidence_hash="sha256:" + "c" * 64,
            maximum_inner_steps=192,
        )


def test_private_fixed_factory_preserves_law_and_rejects_model_or_source_changes(
    request, monkeypatch
):
    from rosclaw_soccer.rsi.extended_proposal_episode_factory import ExtendedProposalEpisodeFactory

    artifact = make_model(parent(request), maximum_inner_steps=1600)
    factory = ExtendedProposalEpisodeFactory(artifact)
    a, b = factory.new_episode(), factory.new_episode()
    query = np.ones(134) * 0.31
    assert np.array_equal(a.raw_mean(query, 1), factory._prototype.raw_mean(query, 1))
    assert a._decoder is not b._decoder
    assert a._memory is not b._memory
    assert a._parent._memory is not b._parent._memory
    a._memory.advance(0, np.zeros(6))
    assert b._memory.last_frame == factory._prototype._memory.last_frame == -1
    assert factory.preview(artifact) == make_preview(artifact)
    forged = copy.deepcopy(artifact)
    forged["maximum_inner_steps"] = 1599
    forged.pop("model_hash")
    forged["model_hash"] = hash_json(forged)
    with pytest.raises(ValueError, match="canonical"):
        factory.preview(forged)
    pins = dict(factory._pins)
    path = next(iter(pins))
    pins[path] = "sha256:" + "f" * 64
    monkeypatch.setattr(factory, "_pins", pins)
    with pytest.raises(ValueError, match="sources"):
        factory.new_episode()
