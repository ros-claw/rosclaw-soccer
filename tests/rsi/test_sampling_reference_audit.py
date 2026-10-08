"""Synthetic seed/template/parity tests, not native G1 qualification."""

import copy
import json

import numpy as np
import pytest

import rosclaw_soccer.rsi.sampling_reference_audit as reference_module
from rosclaw_soccer.rsi.cpu_motor_transfer_evidence import audit_cpu_transfer
from rosclaw_soccer.rsi.recurrent_sampling_motor import (
    TRACE_FIELDS,
    CompiledRecurrentSamplingMotor,
    make_preview,
    make_sampling_view,
)
from rosclaw_soccer.rsi.sampling_reference_audit import SamplingReferenceAuditCompiler
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_recurrent_clipped_motor import learned  # noqa: F401
from tests.rsi.test_recurrent_sampling_motor import long_body
from tests.rsi.test_recurrent_success_motor import boundary
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_two_real_seed_laws_all300_frames_match_original_and_states_stay_owned(learned):  # noqa: F811
    for mean in (learned[0], learned[-1]):
        initial = make_preview(make_sampling_view(mean, seed=771))
        compiler = SamplingReferenceAuditCompiler(initial)
        noises = []
        for seed in (771, 772):
            policy = make_preview(make_sampling_view(mean, seed=seed))
            reference = CompiledRecurrentSamplingMotor(policy)
            actual, untouched = compiler.new_episode(policy), compiler.new_episode(policy)
            noises.append(actual._noise.copy())
            assert not actual._noise.flags.writeable
            assert not np.shares_memory(actual._noise, reference._noise)
            previous = np.zeros(12)
            for frame in range(300):
                step = boundary(frame, previous)
                delta = actual.delta_at_frame(policy, long_body(), **step)
                np.testing.assert_array_equal(
                    delta, reference.delta_at_frame(policy, long_body(), **step)
                )
                np.testing.assert_array_equal(actual.hidden_state, reference.hidden_state)
                assert actual._recurrent.next_index == reference._recurrent.next_index
                for key in TRACE_FIELDS:
                    np.testing.assert_array_equal(
                        actual.sampled_transition[key], reference.sampled_transition[key]
                    )
                previous = delta
            np.testing.assert_array_equal(untouched.hidden_state, np.zeros(64))
            np.testing.assert_array_equal(compiler._prototype.hidden_state, np.zeros(64))
            assert untouched._recurrent.next_index == compiler._prototype._recurrent.next_index == 0
            assert actual._parent is actual._behavior._parent
            assert actual._parent._memory is not untouched._parent._memory
            actual._recurrent._parameters["head_bias"][0] += 0.1
            compiler.verify_policy(policy)
            assert compiler.new_episode(policy)._policy_hash == policy["policy_hash"]
        assert not np.array_equal(*noises)
        contract = compiler.contract()
        assert contract["producer_factory_or_preview_reused"] is False
        assert all(
            contract[k] is False
            for k in ("runtime_execution_authorized", "promotion_authorized", "hardware_authorized")
        )


def test_changed_full_mean_template_noise_law_sources_and_prototype_refused(learned):  # noqa: F811
    original = make_preview(make_sampling_view(learned[-1], seed=771))
    compiler = SamplingReferenceAuditCompiler(original)
    for field in ("critic_parameters", "previous_critic_parameters", "parameters"):
        changed = copy.deepcopy(original)
        parameters = changed["step_motor_proof"]["model"]["mean_model"][field]
        parameters[next(iter(parameters))][0][0] += 0.001
        with pytest.raises(ValueError, match="mean, template"):
            compiler.verify_policy(changed)
    for field, value in (("rho", 0.8), ("std_raw", 0.2), ("source_hash", "changed")):
        changed = copy.deepcopy(original)
        changed["step_motor_proof"]["model"][field] = value
        with pytest.raises(ValueError, match="mean, template"):
            compiler.new_episode(changed)
    changed = copy.deepcopy(original)
    changed["step_motor_proof"]["decision_start_frame"] = 31
    with pytest.raises(ValueError, match="mean, template"):
        compiler.verify_policy(changed)
    for seed in (True, -1, 2**32, 1.0, "771", None):
        changed = copy.deepcopy(original)
        changed["step_motor_proof"]["model"]["seed"] = seed
        with pytest.raises(ValueError, match="integer"):
            compiler.verify_policy(changed)
    compiler._prototype._recurrent._state[0] = 0.1
    with pytest.raises(ValueError, match="prototype"):
        compiler.new_episode(original)


def test_caller_mutation_during_clone_cannot_change_verified_policy_hash(learned, monkeypatch):  # noqa: F811
    policy = make_preview(make_sampling_view(learned[-1], seed=771))
    compiler = SamplingReferenceAuditCompiler(policy)
    verified_hash = policy["policy_hash"]
    original_copy = copy.deepcopy

    def mutate_caller(value, memo=None):
        if type(value) is CompiledRecurrentSamplingMotor:
            policy["policy_hash"] = "unverified-caller-mutation"
        return original_copy(value, memo)

    monkeypatch.setattr(copy, "deepcopy", mutate_caller)
    episode = compiler.new_episode(policy)
    assert policy["policy_hash"] != verified_hash
    assert episode._policy_hash == verified_hash


def test_exact_complete_repeat_skips_only_redundant_hashes(learned, monkeypatch):  # noqa: F811
    policy = make_preview(make_sampling_view(learned[-1], seed=771))
    compiler = SamplingReferenceAuditCompiler(policy)
    independently_owned = copy.deepcopy(policy)
    verified_hash = policy["policy_hash"]

    def unnecessary_hash(_value):
        raise AssertionError("exact whole policy must not re-encode seed-derived hashes")

    def unnecessary_restore(_snapshot):
        raise AssertionError("exact whole policy must not allocate another decoded full model")

    monkeypatch.setattr(reference_module, "hash_json", unnecessary_hash)
    monkeypatch.setattr(reference_module.CanonicalJSONSnapshot, "restore", unnecessary_restore)
    compiler.verify_policy(independently_owned)
    episode = compiler.new_episode(independently_owned)
    assert episode._policy_hash == verified_hash
    assert episode._sampling["seed"] == 771
    assert episode._memory is not compiler._prototype._memory
    assert not np.shares_memory(episode.hidden_state, compiler._prototype.hidden_state)
    # Identity still depends on immutable bytes and the original prototype.
    object.__setattr__(compiler._policy, "_data", b"{}")
    with pytest.raises(ValueError, match="snapshot"):
        compiler.verify_policy(independently_owned)


def test_exact_repeat_still_rejects_private_sampling_seed_mutation(learned):  # noqa: F811
    policy = make_preview(make_sampling_view(learned[-1], seed=771))
    compiler = SamplingReferenceAuditCompiler(policy)
    assert compiler._prototype._sampling is not None
    compiler._prototype._sampling["seed"] = 772
    with pytest.raises(ValueError, match="prototype"):
        compiler.new_episode(copy.deepcopy(policy))


def test_changed_seed_still_restores_both_owned_complete_documents(learned, monkeypatch):  # noqa: F811
    initial = make_preview(make_sampling_view(learned[-1], seed=771))
    compiler = SamplingReferenceAuditCompiler(initial)
    changed = make_preview(make_sampling_view(learned[-1], seed=772))
    original_restore = reference_module.CanonicalJSONSnapshot.restore
    restored_hashes = []

    def tracked_restore(snapshot):
        restored_hashes.append(snapshot.content_hash)
        return original_restore(snapshot)

    monkeypatch.setattr(reference_module.CanonicalJSONSnapshot, "restore", tracked_restore)
    compiler.verify_policy(changed)
    assert restored_hashes == [
        reference_module.CanonicalJSONSnapshot(changed).content_hash,
        compiler._policy.content_hash,
    ]


def test_exact_repeat_never_trusts_only_callers_claimed_policy_hash(learned):  # noqa: F811
    policy = make_preview(make_sampling_view(learned[-1], seed=771))
    compiler = SamplingReferenceAuditCompiler(policy)
    for field, value in (("policy_hash", "forged"), ("unexpected_field", True)):
        changed = copy.deepcopy(policy)
        changed[field] = value
        with pytest.raises(ValueError, match="mean, template"):
            compiler.verify_policy(changed)
    changed = copy.deepcopy(policy)
    changed["step_motor_proof"]["model"]["mean_model"]["parameters"]["head_bias"][0] += 0.01
    # Deliberately keep the original claimed policy/model hashes.
    with pytest.raises(ValueError, match="mean, template"):
        compiler.new_episode(changed)
    changed = copy.deepcopy(policy)
    changed["unexpected_field"] = float("nan")
    with pytest.raises(ValueError):
        compiler.verify_policy(changed)


def test_source_drift_and_caller_ownership(learned, monkeypatch):  # noqa: F811
    original = make_preview(make_sampling_view(learned[-1], seed=771))
    compiler = SamplingReferenceAuditCompiler(original)
    original["step_motor_proof"]["model"]["rho"] = 0.8
    good = make_preview(make_sampling_view(learned[-1], seed=772))
    compiler.verify_policy(good)
    compiler.contract()["source_pins"].clear()
    assert compiler.contract()["source_pins"]
    monkeypatch.setitem(compiler._pins, next(iter(compiler._pins)), "sha256:" + "f" * 64)
    with pytest.raises(ValueError, match="source"):
        compiler.verify_policy(good)


@pytest.mark.parametrize("value", [object(), lambda: None])
def test_external_sampling_reference_callbacks_rejected_before_physics(tmp_path, value):
    report = {"executed_motor_policy": {"recurrent_sampling_motor_proof": {}}}
    report["report_hash"] = hash_json(report)
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="independent source-bound"):
        audit_cpu_transfer(tmp_path, tmp_path / "unopened.py", sampling_reference_compiler=value)


@pytest.mark.parametrize("value", [{}, None, [], {"recurrent_clipped_motor_proof": {}}])
def test_other_families_rejected(value):
    with pytest.raises(ValueError, match="complete recurrent sampling"):
        SamplingReferenceAuditCompiler(value)


def test_valid_compiler_requires_executed_model_binding_and_no_other_factories(learned, tmp_path):  # noqa: F811
    policy = make_preview(make_sampling_view(learned[-1], seed=771))
    compiler = SamplingReferenceAuditCompiler(policy)
    model_hash = policy["step_motor_proof"]["model"]["model_hash"]
    cases = [
        (None, {}),
        ("unbound-model", {}),
        (model_hash, {"sampling_decoder_factory": object()}),
        (model_hash, {"mean_decoder_factory": object()}),
        (model_hash, {"recurrent_reference_compiler": object()}),
    ]
    for executed_hash, extra in cases:
        report = {"executed_motor_policy": policy, "step_model_hash": executed_hash}
        report["report_hash"] = hash_json(report)
        (tmp_path / "report.json").write_text(json.dumps(report))
        with pytest.raises(ValueError, match="exact independent source-bound"):
            audit_cpu_transfer(
                tmp_path,
                tmp_path / "unopened.py",
                sampling_reference_compiler=compiler,
                **extra,
            )
