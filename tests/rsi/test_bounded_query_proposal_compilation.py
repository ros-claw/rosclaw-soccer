import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.bounded_query_proposal_compilation import compile_bounded_query_proposal
from rosclaw_soccer.rsi.proposal_memory_learning import fit_update
from rosclaw_soccer.rsi.proposal_memory_motor import initial_model, make_preview
from rosclaw_soccer.rsi.proposal_snapshot_compilation import compile_proposal_snapshot
from tests.rsi.test_correlated_gradient_numerical_parity import conditional_fixture_batch
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_domain_memory_protection import synthetic_protection
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.mark.parametrize("learned", [False, True])
def test_three_owned_guard_queries_preserve_raw_outputs_and_logical_banks(
    current,  # noqa: F811
    smooth_parent,  # noqa: F811
    learned,
):
    initial = initial_model(current[0], maximum_mean_kl=0.05)
    value = (
        fit_update(
            initial, conditional_fixture_batch(smooth_parent), batch_hash="sha256:" + "a" * 64
        )
        if learned
        else initial
    )
    policy = make_preview(value)
    before = copy.deepcopy(policy)
    reference = compile_proposal_snapshot(policy)
    fast = compile_bounded_query_proposal(policy)
    assert policy == before
    assert fast._policy_hash == reference._policy_hash
    queries = np.vstack(
        (np.zeros((1, 134)), np.ones((1, 134)), np.random.default_rng(485).normal(size=(16, 134)))
    )
    for phase in range(3):
        for query in queries:
            np.testing.assert_array_equal(
                fast.raw_mean(query, phase), reference.raw_mean(query, phase)
            )
    for a, b in (
        (fast._guard, reference._guard),
        (fast._parent._guard, reference._parent._guard),
        (fast._parent._output_memory._guard, reference._parent._output_memory._guard),
    ):
        assert a.to_dict() == b.to_dict()
    assert fast._parent._output_memory.to_dict() == reference._parent._output_memory.to_dict()


def test_multidomain_complete_bank_is_not_replaced_or_evicted(current):  # noqa: F811
    actor, _ = current
    plain = compile_proposal_snapshot(make_preview(initial_model(actor, maximum_mean_kl=0.05)))
    query = np.full(134, 0.73)
    states = np.tile(np.append(plain.features(query)[:134], 1), (270, 1)).tolist()
    protection = synthetic_protection(actor, states)
    policy = make_preview(
        initial_model(actor, maximum_mean_kl=0.05, protected_domain_bank=protection)
    )
    reference = compile_proposal_snapshot(policy)
    fast = compile_bounded_query_proposal(policy)
    assert fast._guard.bank() == reference._guard.bank()
    assert fast._guard.bank_hash == reference._guard.bank_hash
    assert fast._guard.gate(states[0]) == reference._guard.gate(states[0]) == 0.0
