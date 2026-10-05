"""Synthetic sequence contracts; not G1 physics or policy qualification."""

import copy

import numpy as np
import pytest
from rosclaw.growth.causal_residual_memory import initial_parameters
from rosclaw.growth.recurrent_residual_imitation import (
    RecurrentResidualImitationConfig,
    fit_recurrent_residual_imitation,
)

from rosclaw_soccer.rsi.proposal_decoder_selection import select_proposal_decoder
from rosclaw_soccer.rsi.proposal_memory_motor import make_preview as parent_preview
from rosclaw_soccer.rsi.recurrent_success_motor import (
    CompiledRecurrentSuccessMotor,
    make_model,
    make_preview,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_execution import body
from tests.rsi.test_step_motor_network import model  # noqa: F401


def boundary(frame, previous):
    return dict(
        frame=frame,
        nominal_target=np.zeros(29),
        baseline=np.zeros(12),
        limits=np.tile([-1.0, 1.0], (12, 1)),
        previous=previous,
        previous_contact_forces=np.zeros(6),
    )


def reseal(value):
    value.pop("model_hash")
    value["model_hash"] = hash_json(value)
    return value


def test_zero_head_preserves_complete_frozen_boundary_and_owns_state(imitation_parent):  # noqa: F811
    artifact = make_model(imitation_parent, initial_seed=690)
    policy = make_preview(artifact)
    first, second = [CompiledRecurrentSuccessMotor(policy) for _ in range(2)]
    original_policy = parent_preview(imitation_parent)
    original = select_proposal_decoder(original_policy, implementation="bounded_snapshot")
    observation = body()
    previous = np.zeros(12)
    for frame in range(36):
        step = boundary(frame, previous)
        expected = original.delta_at_frame(original_policy, observation, **step)
        actual = first.delta_at_frame(policy, observation, **step)
        np.testing.assert_array_equal(actual, expected)
        assert np.max(np.abs(actual)) <= 0.16
        assert np.max(np.abs(actual - previous)) <= 0.012 + 1e-15
        if frame < 30:
            np.testing.assert_array_equal(first.hidden_state, np.zeros(64))
        previous = actual
    np.testing.assert_array_equal(second.hidden_state, np.zeros(64))
    assert first._memory is not second._memory
    assert first._parent._memory is not second._parent._memory
    assert first._parent._output_memory.to_dict() == original._parent._output_memory.to_dict()
    view = first.hidden_state
    view[:] = 99
    assert np.max(np.abs(first.hidden_state)) <= 1
    artifact["parameters"]["head_bias"][0] = 1
    assert first._recurrent.parameters()["head_bias"][0] == 0


def test_future_rows_do_not_enter_recurrent_state_or_output(imitation_parent):  # noqa: F811
    policy = make_preview(make_model(imitation_parent))
    a, b = [CompiledRecurrentSuccessMotor(policy) for _ in range(2)]
    original, changed = body(), body()
    for name in changed:
        changed[name][32:] = 123
    previous = np.zeros(12)
    for frame in range(32):
        step = boundary(frame, previous)
        left = a.delta_at_frame(policy, original, **step)
        right = b.delta_at_frame(policy, changed, **step)
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(a.hidden_state, b.hidden_state)
        previous = left
    before = a.hidden_state
    with pytest.raises(ValueError, match="sequential motor boundary"):
        a.raw_mean(np.zeros(134), 0)
    np.testing.assert_array_equal(before, a.hidden_state)
    with pytest.raises(ValueError):
        a.delta_at_frame(policy, original, **boundary(31, previous))
    np.testing.assert_array_equal(before, a.hidden_state)


@pytest.mark.parametrize(
    "field",
    [
        "promotion_authorized",
        "hardware_authorized",
        "runtime_execution_authorized",
        "fresh_holdout_open_authorized",
        "physical_action_bounds_changed",
        "protected_memory_changed",
    ],
)
def test_resealed_authority_or_bounds_change_rejected(imitation_parent, field):  # noqa: F811
    value = make_model(imitation_parent)
    value[field] = True
    with pytest.raises(ValueError):
        validate_model(reseal(value))


@pytest.mark.parametrize("field", ["head_bias", "bias_ih", "bias_hh"])
def test_unreceipted_parameter_change_rejected(imitation_parent, field):  # noqa: F811
    value = make_model(imitation_parent)
    value["parameters"][field][0] += 0.01
    with pytest.raises(ValueError, match="unreceipted"):
        validate_model(reseal(value))


def test_source_and_parent_identity_cannot_be_resealed(imitation_parent):  # noqa: F811
    for field in ("source_hash", "memory_source_hash", "learner_source_hash", "parent_model_hash"):
        value = make_model(imitation_parent)
        value[field] = "sha256:" + "f" * 64
        with pytest.raises(ValueError):
            validate_model(reseal(value))
    value = make_model(imitation_parent)
    changed = copy.deepcopy(value["frozen_parent"])
    changed["hardware_authorized"] = True
    value["frozen_parent"] = reseal(changed)
    value["parent_model_hash"] = changed["model_hash"]
    with pytest.raises(ValueError):
        validate_model(reseal(value))


def fitted_synthetic_model(parent):
    original = select_proposal_decoder(parent_preview(parent), implementation="bounded_snapshot")
    query = np.zeros(134)
    context = np.concatenate((original.features(query)[:134], [0]))
    x = np.broadcast_to(context, (4, 270, 135)).copy()
    base = np.broadcast_to(original.raw_mean(query, 0), (4, 270, 12)).copy()
    gates = np.full((4, 270), original._guard.gate(context))
    assert gates[0, 0] > 0
    result = fit_recurrent_residual_imitation(
        parameters=initial_parameters(135, 12, hidden_dimension=64, seed=19),
        context=x,
        baseline=base,
        gates=gates,
        targets=base + 0.01,
        training_weights=np.ones((4, 270)),
        config=RecurrentResidualImitationConfig(steps=4, seed=19),
    )
    receipt = {k: v for k, v in result.items() if k != "parameters"}
    receipt.update(
        behavior_model_hash=parent["model_hash"],
        physical_batch_hash="sha256:" + "b" * 64,
        teacher_selection_hash="sha256:" + "c" * 64,
        teacher_selection_offline=True,
        all_success_and_failure_episodes_retained=True,
        teacher_labels_are_actor_inputs=False,
        private_fresh_accessed=False,
    )
    value = make_model(
        parent,
        initial_seed=19,
        parameters=result["parameters"],
        learning_receipt=receipt,
    )
    return value


def test_actual_sequence_fit_receipt_is_bound_but_not_physical_evidence(imitation_parent):  # noqa: F811
    value = fitted_synthetic_model(imitation_parent)
    assert value["learning_receipt"]["physical_batch_verified"] is False
    assert value["learning_receipt"]["input_feature_causality_verified"] is False
    assert value["promotion_authorized"] is False
    policy = make_preview(value)
    decoder = CompiledRecurrentSuccessMotor(policy)
    previous = np.zeros(12)
    observation = body()
    for frame in range(35):
        previous = decoder.delta_at_frame(policy, observation, **boundary(frame, previous))
    assert np.max(np.abs(decoder.hidden_state)) <= 1
    for fault in ("fitted_parameters_hash", "behavior_model_hash", "physical_batch_verified"):
        changed = copy.deepcopy(value)
        changed["learning_receipt"][fault] = True if fault.endswith("verified") else "bad"
        with pytest.raises(ValueError, match="receipt"):
            validate_model(reseal(changed))
