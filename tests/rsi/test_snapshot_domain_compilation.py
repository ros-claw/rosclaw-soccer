"""Synthetic storage/gate/decoder parity, no native G1 performance claim."""

import copy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from rosclaw.growth.bounded_query_anchor_guard import BoundedQueryDomainAnchorGuard
from rosclaw.growth.domain_anchor_bank import DomainAnchorGuard, build_domain_anchor_bank

from rosclaw_soccer.rsi.fixed_recurrent_reference_audit import _numeric_graph_hash
from rosclaw_soccer.rsi.recurrent_sampling_motor import (
    CompiledRecurrentSamplingMotor,
    make_preview,
    make_sampling_view,
)
from rosclaw_soccer.rsi.recurrent_success_motor import make_model
from rosclaw_soccer.rsi.snapshot_domain_compilation import (
    SnapshotDomainAnchorGuard,
    SnapshotRecurrentSamplingEpisodeFactory,
    compile_snapshot_banks,
    restore_original_banks,
)
from tests.rsi.test_recurrent_sampling_episode_factory import (  # noqa: F401
    candidate,
    current,
    imitation_parent,
    model,
    smooth_parent,
)
from tests.rsi.test_recurrent_sampling_motor import long_body
from tests.rsi.test_recurrent_success_motor import boundary


def guard(cls=DomainAnchorGuard):
    bank = build_domain_anchor_bank(
        [
            dict(
                domain_id="toy",
                source_evidence_hash="sha256:" + "1" * 64,
                context_ids=["a", "b"],
                context_rows=[2, 1],
                observations=[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            )
        ],
        parent_hash="sha256:" + "2" * 64,
        encoder_hash="sha256:" + "3" * 64,
    )
    return cls(bank)


@pytest.mark.parametrize("cls", [DomainAnchorGuard, BoundedQueryDomainAnchorGuard])
def test_complete_bank_gate_metric_and_immutable_serialization(cls):
    source = guard(cls)
    compiled = SnapshotDomainAnchorGuard(source)
    assert compiled.bank() == source.bank() and compiled.to_dict() == source.to_dict()
    points = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [-0.001, 0.001], [2.0, 2.0], [5.0, 5.0]])
    np.testing.assert_array_equal(compiled.gates(points), source.gates(points))
    np.testing.assert_array_equal(copy.deepcopy(compiled).gates(points), source.gates(points))
    assert _numeric_graph_hash(compiled) == _numeric_graph_hash(copy.deepcopy(compiled))
    restored = compiled.original_storage()
    assert type(restored) is cls and _numeric_graph_hash(restored) == _numeric_graph_hash(source)
    changed = compiled.bank()
    changed["domains"][0]["observations"][0][0] = 55.0
    assert compiled.bank() == source.bank()
    with pytest.raises(FrozenInstanceError):
        compiled._bank_snapshot._data += b" "
    # Deliberate Python object-model bypass must still fail full-byte verify.
    object.__setattr__(compiled._bank_snapshot, "_data", compiled._bank_snapshot._data + b" ")
    with pytest.raises(ValueError):
        compiled.bank()


@pytest.mark.parametrize("cls", [DomainAnchorGuard, BoundedQueryDomainAnchorGuard])
def test_whole_graph_aliases_and_state_reversible(monkeypatch, request, cls):
    mean = make_model(request.getfixturevalue("imitation_parent"))
    policy = make_preview(make_sampling_view(mean, seed=901))
    decoder = CompiledRecurrentSamplingMotor(policy)
    # Explicit test-only metadata nodes. This is not an admissible original
    # policy declaration or proof about a real model's active domain guards.
    decoder.test_guard = guard(cls)
    decoder.test_guard_alias = decoder.test_guard
    original = _numeric_graph_hash(decoder)
    compiled, review = compile_snapshot_banks(decoder)
    assert (
        review["converted_guard_count"] >= 1
        and review["reversible_whole_graph_and_aliases_verified"]
    )
    assert compiled.test_guard is compiled.test_guard_alias
    assert type(compiled.test_guard) is SnapshotDomainAnchorGuard
    assert _numeric_graph_hash(decoder) == original
    assert compiled._memory is not decoder._memory and compiled._recurrent is not decoder._recurrent
    previous = np.zeros(12)
    body = long_body()
    for frame in range(300):
        kwargs = boundary(frame, previous)
        a = decoder.delta_at_frame(policy, body, **kwargs)
        b = compiled.delta_at_frame(policy, body, **kwargs)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(decoder.hidden_state, compiled.hidden_state)
        previous = a
    assert _numeric_graph_hash(restore_original_banks(compiled)) == _numeric_graph_hash(decoder)
    with pytest.raises(ValueError):
        compile_snapshot_banks(compiled)


def test_factory_full_original_allocation_and_fresh_episodes(request):
    mean = make_model(request.getfixturevalue("imitation_parent"))
    factory = SnapshotRecurrentSamplingEpisodeFactory(mean)
    first, decoder = factory.generate_and_bind_seed(seed=773)
    reference = CompiledRecurrentSamplingMotor(first)
    other_policy, other = factory.generate_and_bind_seed(seed=773)
    assert first == other_policy
    previous = np.zeros(12)
    body = long_body()
    for frame in range(300):
        kwargs = boundary(frame, previous)
        left = decoder.delta_at_frame(first, body, **kwargs)
        right = reference.delta_at_frame(first, body, **kwargs)
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(decoder.hidden_state, reference.hidden_state)
        previous = left
    assert other._recurrent.next_index == 0 and np.all(other.hidden_state == 0)
    contract = factory.contract()
    assert (
        contract["complete_original_constructor_executed"]
        and not contract["physical_parity_verified"]
    )
    assert not contract["automatically_selected_by_native_collector"]
    factory._prototype._last_draw[0] = 0.1
    with pytest.raises(ValueError):
        factory.generate_and_bind_seed(seed=774)
