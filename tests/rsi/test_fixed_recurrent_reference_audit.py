"""Synthetic compiler ownership/admission, not native or policy qualification."""

import copy
import json

import numpy as np
import pytest

from rosclaw_soccer.rsi.cpu_motor_transfer_evidence import audit_cpu_transfer
from rosclaw_soccer.rsi.fixed_recurrent_reference_audit import (
    FixedRecurrentReferenceAuditCompiler,
    _numeric_graph_hash,
)
from rosclaw_soccer.rsi.recurrent_clipped_motor import CompiledRecurrentClippedMotor, make_preview
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_recurrent_clipped_motor import learned  # noqa: F401
from tests.rsi.test_recurrent_sampling_motor import long_body
from tests.rsi.test_recurrent_success_motor import boundary
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_complete300_parity_deep_ownership_and_unadvanced_prototype(learned):  # noqa: F811
    policy = make_preview(learned[-1])
    compiler = FixedRecurrentReferenceAuditCompiler(policy)
    reference = CompiledRecurrentClippedMotor(policy)
    actual, untouched = compiler.new_episode(), compiler.new_episode()
    previous = np.zeros(12)
    trace = long_body()
    for frame in range(300):
        step = boundary(frame, previous)
        delta = actual.delta_at_frame(policy, trace, **step)
        np.testing.assert_array_equal(delta, reference.delta_at_frame(policy, trace, **step))
        np.testing.assert_array_equal(actual.hidden_state, reference.hidden_state)
        assert actual._recurrent.next_index == reference._recurrent.next_index
        previous = delta
    assert compiler._prototype._active_frame is None
    assert untouched._active_frame is None
    np.testing.assert_array_equal(untouched.hidden_state, np.zeros(64))
    assert untouched._recurrent.next_index == 0
    assert actual._memory is not untouched._memory
    assert actual._behavior._memory is not untouched._behavior._memory
    assert actual._parent is actual._behavior._parent
    assert actual._parent is not untouched._parent
    assert not np.shares_memory(
        actual._recurrent._parameters["weight_hh"], untouched._recurrent._parameters["weight_hh"]
    )
    actual._recurrent._parameters["head_bias"][0] += 1
    np.testing.assert_array_equal(compiler.new_episode().hidden_state, np.zeros(64))
    compiler.verify_policy(policy)
    contract = compiler.contract()
    assert contract["producer_factory_reused"] is False
    assert contract["physics_parity_requires_external_evidence"] is True
    assert all(
        contract[k] is False
        for k in ("runtime_execution_authorized", "promotion_authorized", "hardware_authorized")
    )
    contract["source_pins"].clear()
    assert compiler.contract()["source_pins"]


def test_whole_policy_including_critic_and_source_drift(learned, monkeypatch):  # noqa: F811
    policy = make_preview(learned[-1])
    compiler = FixedRecurrentReferenceAuditCompiler(policy)
    for key in ("critic_parameters", "previous_critic_parameters", "parameters"):
        changed = copy.deepcopy(policy)
        weights = changed["step_motor_proof"]["model"][key]
        weights[next(iter(weights))][0][0] += 0.001
        with pytest.raises(ValueError, match="complete fixed"):
            compiler.verify_policy(changed)
    policy["step_motor_proof"]["decision_start_frame"] = 29
    with pytest.raises(ValueError, match="complete fixed"):
        compiler.verify_policy(policy)
    path = next(iter(compiler._pins))
    monkeypatch.setitem(compiler._pins, path, "sha256:" + "f" * 64)
    with pytest.raises(ValueError, match="source"):
        compiler.new_episode()


def test_prototype_numeric_or_history_drift_refused(learned):  # noqa: F811
    compiler = FixedRecurrentReferenceAuditCompiler(make_preview(learned[-1]))
    compiler._prototype._recurrent._state[0] = 0.1
    with pytest.raises(ValueError, match="prototype"):
        compiler.new_episode()


def test_valid_compiler_cannot_certify_unbound_or_absent_executed_model(learned, tmp_path):  # noqa: F811
    policy = make_preview(learned[-1])
    compiler = FixedRecurrentReferenceAuditCompiler(policy)
    for claimed in (None, "sha256:" + "0" * 64):
        value = {"executed_motor_policy": policy, "step_model_hash": claimed}
        value["report_hash"] = hash_json(value)
        (tmp_path / "report.json").write_text(json.dumps(value))
        with pytest.raises(ValueError, match="independent deterministic"):
            audit_cpu_transfer(
                tmp_path, tmp_path / "never-opened.py", recurrent_reference_compiler=compiler
            )


@pytest.mark.parametrize("policy", [{}, {"recurrent_sampling_motor_proof": {}}, [], None])
def test_no_other_policy_family(policy):
    with pytest.raises(ValueError, match="deterministic"):
        FixedRecurrentReferenceAuditCompiler(policy)


@pytest.mark.parametrize("factory", [object(), lambda: None])
def test_external_compiler_callbacks_refused_before_physics(tmp_path, factory):
    value = {"executed_motor_policy": {"recurrent_clipped_motor_proof": {}}}
    value["report_hash"] = hash_json(value)
    (tmp_path / "report.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="independent deterministic"):
        audit_cpu_transfer(
            tmp_path, tmp_path / "never-opened.py", recurrent_reference_compiler=factory
        )


def test_graph_binding_aliases_arrays_bounds_and_unsupported_values():
    a = np.zeros(2)
    assert _numeric_graph_hash({"a": a, "b": a}) != _numeric_graph_hash({"a": a, "b": a.copy()})
    value = {"a": a, "b": a}
    assert _numeric_graph_hash(value) == _numeric_graph_hash(copy.deepcopy(value))
    a[0] = 1
    assert _numeric_graph_hash(value) != _numeric_graph_hash({"a": np.zeros(2), "b": np.zeros(2)})
    for bad in (object(), np.array([np.nan]), {1: "bad"}, np.array(["bad"])):
        with pytest.raises(ValueError):
            _numeric_graph_hash(bad)
    deep = []
    for _ in range(130):
        deep = [deep]
    with pytest.raises(ValueError, match="bounded"):
        _numeric_graph_hash(deep)


def test_spatial_index_full_tree_clone_identity_and_numeric_drift():
    spatial = pytest.importorskip("scipy.spatial")
    tree = spatial.cKDTree(np.arange(80, dtype=float).reshape(40, 2), copy_data=True)
    before = _numeric_graph_hash(tree)
    cloned = copy.deepcopy(tree)
    assert _numeric_graph_hash(cloned) == before
    assert not np.shares_memory(tree.data, cloned.data)
    cloned.data[0, 0] += 1
    assert _numeric_graph_hash(cloned) != before
    assert _numeric_graph_hash(tree) == before


def test_complete_large_json_subtree_binding_is_compact_and_type_sensitive():
    value = {"schema": "fixture.large_model.v1", "weights": [[0.1] * 1024 for _ in range(128)]}
    before = _numeric_graph_hash(value)
    changed = copy.deepcopy(value)
    changed["weights"][-1][-1] += 0.0001
    assert _numeric_graph_hash(changed) != before
    assert _numeric_graph_hash(copy.deepcopy(value)) == before
    assert _numeric_graph_hash({"schema": "fixture", "value": [1]}) != _numeric_graph_hash(
        {"schema": "fixture", "value": [1.0]}
    )
    assert _numeric_graph_hash({"schema": "fixture", "value": [True]}) != _numeric_graph_hash(
        {"schema": "fixture", "value": [1]}
    )
    with pytest.raises(ValueError):
        _numeric_graph_hash({"schema": "fixture", "value": [float("nan")]})
