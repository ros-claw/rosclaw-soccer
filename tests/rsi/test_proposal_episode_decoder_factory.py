"""Synthetic episode-isolation contracts; not physical skill evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.proposal_episode_decoder_factory import ProposalEpisodeDecoderFactory
from rosclaw_soccer.rsi.proposal_memory_learning import fit_update
from rosclaw_soccer.rsi.proposal_memory_motor import (
    CompiledProposalMemoryMotor,
    initial_model,
    make_preview,
)
from tests.rsi.test_correlated_gradient_numerical_parity import conditional_fixture_batch
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.mark.parametrize("learned", [False, True])
def test_episode_parameters_exact_and_histories_independent(
    current,  # noqa: F811
    smooth_parent,  # noqa: F811
    learned,
):
    initial = initial_model(current[0], maximum_mean_kl=0.05)
    proposed = (
        fit_update(
            initial, conditional_fixture_batch(smooth_parent), batch_hash="sha256:" + "a" * 64
        )
        if learned
        else initial
    )
    policy = make_preview(proposed)
    before = copy.deepcopy(policy)
    reference = CompiledProposalMemoryMotor(policy)
    factory = ProposalEpisodeDecoderFactory(policy)
    first, second = factory.new_episode(), factory.new_episode()
    assert policy == before
    assert factory.policy_hash == first._policy_hash == second._policy_hash == policy["policy_hash"]
    assert first is not second and first._parent is not second._parent
    assert first._parent._warm is not second._parent._warm
    assert first._layers is not second._layers
    assert first._parent._layers is not second._parent._layers
    assert first._parent._residual_layers is not second._parent._residual_layers
    assert first._parent._warm.layers is not second._parent._warm.layers
    for query in np.random.default_rng(439).normal(size=(8, 134)):
        np.testing.assert_array_equal(first.features(query), reference.features(query))
        for phase in range(3):
            np.testing.assert_array_equal(
                first.raw_mean(query, phase), reference.raw_mean(query, phase)
            )
            np.testing.assert_array_equal(
                second.raw_mean(query, phase), reference.raw_mean(query, phase)
            )
    assert first._parent._output_memory.to_dict() == reference._parent._output_memory.to_dict()
    assert first._guard.to_dict() == reference._guard.to_dict()
    for memory in (first._memory, first._parent._memory):
        assert memory.advance(0, np.zeros(6)) == 0
        assert memory.advance(1, np.array([2, 0, 0, 0, 0, 0])) == 1
    for memory in (second._memory, second._parent._memory):
        assert memory.last_frame == -1 and memory.first_contact_frame is None
        assert memory.advance(0, np.zeros(6)) == 0
        assert memory.advance(1, np.zeros(6)) == 0
    third = factory.new_episode()
    assert third._memory.last_frame == third._parent._memory.last_frame == -1
    assert first._memory.first_contact_frame == 0
    assert second._memory.first_contact_frame is None
    assert first._layers[0][0] is second._layers[0][0]
    with pytest.raises(ValueError):
        first._layers[0][0][0, 0] = 99
    first._layers.clear()
    assert len(second._layers) == len(third._layers) == 3
    policy["step_motor_proof"]["model"]["residual_layers"][-1]["bias"][0] += 100
    query = np.zeros(134)
    np.testing.assert_array_equal(third.raw_mean(query, 0), reference.raw_mean(query, 0))
    assert first._sampling is first._noise is second._sampling is second._noise is None


def test_invalid_policy_rejected_before_any_episode(current):  # noqa: F811
    policy = make_preview(initial_model(current[0], maximum_mean_kl=0.05))
    policy["step_motor_proof"]["promotion_authorized"] = True
    with pytest.raises(ValueError):
        ProposalEpisodeDecoderFactory(policy)


def test_private_factory_rejects_source_changes(current, monkeypatch):  # noqa: F811
    from rosclaw_soccer.rsi import proposal_episode_decoder_factory as module

    factory = ProposalEpisodeDecoderFactory(
        make_preview(initial_model(current[0], maximum_mean_kl=0.05))
    )
    contract = factory.contract()
    assert contract["complete_original_preview_validation_at_allocation"] is True
    assert contract["hardware_authorized"] is False
    monkeypatch.setattr(module, "hash_bytes", lambda _: "source changed")
    for operation in (factory.new_episode, factory.contract, lambda: factory.policy_hash):
        with pytest.raises(ValueError, match="sources changed"):
            operation()
