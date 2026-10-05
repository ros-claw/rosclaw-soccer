"""Actual student sampling semantics; synthetic contracts, not G1 evidence."""

import copy

import numpy as np
import pytest
from rosclaw.growth.causal_residual_memory import CausalResidualMemory
from rosclaw.growth.staged_action_projection import staged_action_projection

from rosclaw_soccer.rsi.recurrent_sampling_motor import (
    TRACE_FIELDS,
    CompiledRecurrentSamplingMotor,
    make_preview,
    make_sampling_view,
)
from rosclaw_soccer.rsi.recurrent_success_motor import make_model
from rosclaw_soccer.rsi.stochastic_step_execution import features_at_frame
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_recurrent_success_motor import boundary, fitted_synthetic_model
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_execution import body
from tests.rsi.test_step_motor_network import model  # noqa: F401


def reseal(value):
    value.pop("model_hash")
    value["model_hash"] = hash_json(value)
    return value


def long_body():
    return {k: np.repeat(v[:1], 300, axis=0) for k, v in body().items()}


@pytest.mark.parametrize("trained", [False, True])
def test_full_causal_scan_and_independent_ar_density(imitation_parent, trained):  # noqa: F811
    mean_model = (
        fitted_synthetic_model(imitation_parent) if trained else make_model(imitation_parent)
    )
    view = make_sampling_view(mean_model, seed=767)
    policy = make_preview(view)
    decoder = CompiledRecurrentSamplingMotor(policy)
    reference_memory = CausalResidualMemory(mean_model["parameters"])
    observation, previous, previous_offset = long_body(), np.zeros(12), np.zeros(12)
    for frame in range(300):
        options = boundary(frame, previous)
        if frame >= 30:
            x = features_at_frame(
                observation,
                frame=frame,
                nominal_target=options["nominal_target"],
                previous=previous,
                previous_contact_forces=options["previous_contact_forces"],
            )
            context = np.concatenate((decoder._behavior.features(x)[:134], [0]))
            residual = reference_memory.step(context, index=frame - 30)
            expected_mean = decoder._behavior.raw_mean(x, 0) + (
                0.2 * decoder._guard.gate(context) * residual
            )
            innovation = np.random.default_rng(
                np.random.SeedSequence(767, spawn_key=(frame,))
            ).normal(size=12)
            scale = 0.1 if frame == 30 else 0.1 * np.sqrt(1 - 0.9**2)
            offset = scale * innovation + (np.zeros(12) if frame == 30 else 0.9 * previous_offset)
            expected_latent = expected_mean + offset
            expected_logp = float(
                np.sum(-0.5 * innovation**2 - np.log(scale) - 0.5 * np.log(2 * np.pi))
            )
        actual = decoder.delta_at_frame(policy, observation, **options)
        transition = decoder.sampled_transition
        assert set(transition) == set(TRACE_FIELDS)
        if frame < 30:
            np.testing.assert_array_equal(actual, np.zeros(12))
            assert not transition[TRACE_FIELDS[-1]][0]
            assert all(not np.any(v) for v in transition.values())
            np.testing.assert_array_equal(decoder.hidden_state, np.zeros(64))
        else:
            np.testing.assert_allclose(
                transition[TRACE_FIELDS[0]], expected_mean, atol=1e-14, rtol=0
            )
            np.testing.assert_allclose(
                transition[TRACE_FIELDS[1]], expected_latent, atol=1e-14, rtol=0
            )
            assert transition[TRACE_FIELDS[2]][0] == pytest.approx(expected_logp, abs=1e-12)
            assert transition[TRACE_FIELDS[3]][0]
            expected_delta = staged_action_projection(
                transition[TRACE_FIELDS[1]],
                previous,
                np.full(12, -0.16),
                np.full(12, 0.16),
                cap=0.16,
                slew=0.012,
            )
            np.testing.assert_array_equal(actual, expected_delta)
            np.testing.assert_array_equal(decoder.hidden_state, reference_memory.state)
            assert decoder._recurrent.next_index == frame - 29
            previous_offset = offset
        assert np.max(np.abs(actual)) <= 0.16
        assert np.max(np.abs(actual - previous)) <= 0.012 + 1e-15
        previous = actual
    assert not decoder._noise.flags.writeable
    assert (
        policy["recurrent_sampling_motor_proof"]["actual_behavior_model_hash"]
        == mean_model["model_hash"]
    )
    assert policy["recurrent_sampling_motor_proof"][
        "latent_likelihood_is_not_projected_action_likelihood"
    ]


def test_owned_transitions_and_no_out_of_boundary_state_advance(imitation_parent):  # noqa: F811
    mean = make_model(imitation_parent)
    view = make_sampling_view(mean, seed=768)
    policy = make_preview(view)
    first, second = [CompiledRecurrentSamplingMotor(policy) for _ in range(2)]
    changed, original = body(), body()
    for array in changed.values():
        array[32:] = 123
    previous = np.zeros(12)
    for frame in range(32):
        left = first.delta_at_frame(policy, original, **boundary(frame, previous))
        right = second.delta_at_frame(policy, changed, **boundary(frame, previous))
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(first.hidden_state, second.hidden_state)
        for key in TRACE_FIELDS:
            np.testing.assert_array_equal(
                first.sampled_transition[key], second.sampled_transition[key]
            )
        previous = left
    state = first.hidden_state
    for frame in (True, 29, 30, 31, 300):
        with pytest.raises(ValueError, match="boundary"):
            first.latent_sample(np.zeros(134), frame, 0)
        np.testing.assert_array_equal(first.hidden_state, state)
    returned = first.sampled_transition
    returned[TRACE_FIELDS[0]][:] = 123
    returned[TRACE_FIELDS[-1]][:] = False
    assert np.max(np.abs(first.sampled_transition[TRACE_FIELDS[0]])) < 123
    assert first.sampled_transition[TRACE_FIELDS[-1]][0]
    mean["parameters"]["head_bias"][0] = 1
    assert first._recurrent.parameters()["head_bias"][0] == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("std_raw", 0.15),
        ("rho", 0.8),
        ("seed", True),
        ("training_only", 1),
        ("hardware_authorized", True),
        ("hardware_authorized", 0),
        ("promotion_authorized", True),
        ("runtime_execution_authorized", True),
        ("fresh_holdout_open_authorized", True),
        ("activation_ceiling", "REAL"),
        ("source_hash", "sha256:" + "f" * 64),
        ("exploration_source_hash", "sha256:" + "f" * 64),
        ("extra", True),
    ],
)
def test_resealed_sampling_law_or_authority_change_rejected(imitation_parent, key, value):  # noqa: F811
    view = make_sampling_view(make_model(imitation_parent), seed=767)
    forged = copy.deepcopy(view)
    forged[key] = value
    with pytest.raises(ValueError):
        make_preview(reseal(forged))


def test_changed_mean_or_policy_rejected(imitation_parent):  # noqa: F811
    view = make_sampling_view(make_model(imitation_parent), seed=767)
    forged = copy.deepcopy(view)
    forged["mean_model"]["parameters"]["head_bias"][0] = 0.01
    forged["mean_model"] = reseal(forged["mean_model"])
    with pytest.raises(ValueError):
        make_preview(reseal(forged))
    policy = make_preview(view)
    policy["step_motor_proof"]["decision_start_frame"] = 29
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    with pytest.raises(ValueError, match="commitment"):
        CompiledRecurrentSamplingMotor(policy)
