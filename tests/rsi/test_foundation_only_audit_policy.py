"""Null policy is valid only for an explicitly non-learned foundation run."""

import pytest

from rosclaw_soccer.rsi.cpu_motor_transfer_evidence import _execution_policy


def foundation_report():
    return dict(
        executed_motor_policy=None,
        execution_profile="foundation_only",
        model_hash=None,
        step_model_hash=None,
        motor_policy_hash=None,
        taskspace_gate_selected=False,
    )


def test_explicit_pure_foundation_null_policy_is_not_a_learned_family():
    original = foundation_report()
    assert _execution_policy(original) == {}
    assert original == foundation_report()


def test_original_dict_identity_and_absent_legacy_fixture_unchanged():
    policy = {"recurrent_sampling_motor_proof": {"unchanged": True}}
    assert _execution_policy({"executed_motor_policy": policy}) is policy
    assert _execution_policy({}) == {}


@pytest.mark.parametrize("key", ["model_hash", "step_model_hash", "motor_policy_hash"])
@pytest.mark.parametrize("mutation", ["missing", "learned", "false"])
def test_null_policy_cannot_hide_a_missing_or_learned_motor_identity(key, mutation):
    report = foundation_report()
    if mutation == "missing":
        del report[key]
    else:
        report[key] = False if mutation == "false" else "sha256:" + "a" * 64
    with pytest.raises(ValueError, match="pure foundation-only"):
        _execution_policy(report)


@pytest.mark.parametrize("profile", [None, "", "taskspace_plus_motor", True])
def test_null_policy_requires_exact_foundation_profile(profile):
    report = dict(foundation_report(), execution_profile=profile)
    with pytest.raises(ValueError, match="pure foundation-only"):
        _execution_policy(report)


@pytest.mark.parametrize("gate", [None, 0, True, "false"])
def test_foundation_null_requires_false_taskspace_gate(gate):
    with pytest.raises(ValueError, match="pure foundation-only"):
        _execution_policy(dict(foundation_report(), taskspace_gate_selected=gate))


@pytest.mark.parametrize(
    "key",
    [
        "body_response_guidance",
        "numeric_compilation",
        "numeric_sampling_compilation",
        "recurrent_success_factory",
        "recurrent_sampling_factory",
        "recurrent_clipped_factory",
        "fixed_proposal_factory",
        "extended_proposal_factory",
        "imitation_proposal_factory",
    ],
)
def test_foundation_null_cannot_hide_a_decoder_or_guidance_even_as_null(key):
    report = foundation_report()
    report[key] = None
    with pytest.raises(ValueError, match="pure foundation-only"):
        _execution_policy(report)


@pytest.mark.parametrize("policy", [False, [], "", 0, 1.0])
def test_malformed_policy_is_never_silently_normalized(policy):
    with pytest.raises(ValueError, match="dictionary"):
        _execution_policy(dict(foundation_report(), executed_motor_policy=policy))
