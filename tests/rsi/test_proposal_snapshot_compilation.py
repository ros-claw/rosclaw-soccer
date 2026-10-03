"""Offline compilation contracts, not robot capability evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.proposal_memory_learning import fit_update
from rosclaw_soccer.rsi.proposal_memory_motor import (
    CompiledProposalMemoryMotor,
    initial_model,
    make_preview,
)
from rosclaw_soccer.rsi.proposal_snapshot_compilation import compile_proposal_snapshot
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_correlated_gradient_numerical_parity import conditional_fixture_batch
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_domain_memory_protection import synthetic_protection
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.mark.parametrize("learned", [False, True])
def test_complete_numeric_compilation_matches_reference_and_owns_storage(
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
    fast = compile_proposal_snapshot(policy)
    assert policy == before
    queries = np.concatenate(
        (np.zeros((1, 134)), np.ones((1, 134)), np.random.default_rng(436).normal(size=(24, 134)))
    )
    for phase in range(3):
        for query in queries:
            np.testing.assert_array_equal(fast.features(query), reference.features(query))
            np.testing.assert_array_equal(
                fast.raw_mean(query, phase), reference.raw_mean(query, phase)
            )
    assert fast._parent._output_memory.to_dict() == reference._parent._output_memory.to_dict()
    assert fast._guard.to_dict() == reference._guard.to_dict()
    assert fast._parent._policy_hash == reference._parent._policy_hash
    assert fast._policy_hash == reference._policy_hash
    assert fast._sampling is fast._noise is None
    output_before = fast.raw_mean(queries[0], 0).copy()
    policy["step_motor_proof"]["model"]["residual_layers"][-1]["bias"][0] += 100
    np.testing.assert_array_equal(fast.raw_mean(queries[0], 0), output_before)
    for array in (
        fast._parent._mean,
        fast._parent._scale,
        fast._parent._head,
        *fast._parent._random,
        *(a for pair in fast._layers for a in pair),
        *(a for pair in fast._parent._layers for a in pair),
        *(a for pair in fast._parent._residual_layers for a in pair),
    ):
        assert not array.flags.writeable
    with pytest.raises(ValueError, match="stationary exploration"):
        fast.latent_sample(queries[0], 30, 0)


def test_resealed_source_and_policy_tampering_rejected_before_numeric_allocation(
    current,  # noqa: F811
    monkeypatch,
):
    policy = make_preview(initial_model(current[0], maximum_mean_kl=0.05))
    monkeypatch.setattr(
        "rosclaw_soccer.rsi.proposal_snapshot_compilation._allocate_output_decoder",
        lambda: pytest.fail("invalid snapshot must not allocate numeric decoder"),
    )
    forged = copy.deepcopy(policy)
    forged["step_motor_proof"]["model"]["source_hash"] = "sha256:" + "f" * 64
    inner = forged["step_motor_proof"]["model"]
    inner.pop("model_hash")
    inner["model_hash"] = hash_json(inner)
    forged.pop("policy_hash")
    forged["policy_hash"] = hash_json(forged)
    with pytest.raises(ValueError):
        compile_proposal_snapshot(forged)
    policy["step_motor_proof"]["promotion_authorized"] = True
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    with pytest.raises(ValueError, match="sealed policy"):
        compile_proposal_snapshot(policy)


def test_complete_multidomain_guard_and_bank_are_retained(current):  # noqa: F811
    actor, _ = current
    plain = compile_proposal_snapshot(make_preview(initial_model(actor, maximum_mean_kl=0.05)))
    query = np.full(134, 0.73)
    states = np.tile(np.append(plain.features(query)[:134], 1), (270, 1)).tolist()
    bundle = synthetic_protection(actor, states)
    policy = make_preview(initial_model(actor, maximum_mean_kl=0.05, protected_domain_bank=bundle))
    reference = CompiledProposalMemoryMotor(policy)
    fast = compile_proposal_snapshot(policy)
    assert fast._guard.bank() == reference._guard.bank()
    assert fast._guard.bank_hash == reference._guard.bank_hash
    assert fast._guard.gate(states[0]) == reference._guard.gate(states[0]) == 0
    for phase in range(3):
        np.testing.assert_array_equal(fast.raw_mean(query, phase), reference.raw_mean(query, phase))
