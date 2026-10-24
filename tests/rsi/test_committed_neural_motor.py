import copy
from dataclasses import FrozenInstanceError

import pytest
import torch
from test_refinement_motor_adapter import parent_model

from rosclaw_soccer.rsi.committed_neural_motor import CommittedNeuralMotor
from rosclaw_soccer.rsi.motor_option_state import MotorOptionState
from rosclaw_soccer.rsi.neural_option_initiation import NeuralOptionInitiation
from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter


def fixture():
    p = parent_model().eval().requires_grad_(False)
    experts = {i: RefinementMotorAdapter(p) for i in (1, 2, 3)}
    with torch.no_grad():
        for i, e in experts.items():
            e.head[-1].bias.fill_(i * 0.3)
    a = torch.zeros(1, 1, 160)
    b = torch.ones_like(a) * 0.1
    with torch.no_grad():
        _, _, ea = p.forward_with_evidence(a, p.initial_state())
        _, _, eb = p.forward_with_evidence(b, p.initial_state())
    route = NeuralOptionInitiation(
        torch.cat([ea["critic_features"][0], eb["critic_features"][0]]), torch.tensor([1, 0])
    )
    return (
        p,
        experts,
        CommittedNeuralMotor(p, experts, route, generation_hash="sha256:" + "a" * 64),
        a,
        b,
    )


def test_chosen_neural_skill_is_not_abandoned_when_state_leaves_initiation_region():
    _, experts, motor, a, b = fixture()
    state = motor.initial_state()
    expert_state = experts[1].initial_state()
    actual, state, e = motor.forward_with_evidence(a, state)
    with torch.no_grad():
        expected, expert_state = experts[1](a, expert_state)
    assert state.option == 1 and state.controls == 1 and torch.equal(actual, expected)

    def must_not_reselect(features):
        raise AssertionError("mid-skill reselection")

    motor.initiation.forward = must_not_reselect
    for _ in range(5):
        actual, state, e = motor.forward_with_evidence(b, state)
        with torch.no_grad():
            expected, expert_state = experts[1](b, expert_state)
        assert torch.equal(actual, expected)
        assert state.option == 1 and e["option_initiated"] is False
        assert all(torch.equal(x, y) for x, y in zip(state.recurrent, expert_state, strict=True))
        assert e["final_mean_delta_radians"].abs().max() <= 0.2


def test_two_players_keep_separate_commitments_and_parent_fallback_is_exact():
    parent, _, motor, a, b = fixture()
    sa, sb = motor.initial_state(), motor.initial_state()
    assert sa.recurrent[0].data_ptr() != sb.recurrent[0].data_ptr()
    _, sa, _ = motor.forward_with_evidence(a, sa)
    actual, sb, _ = motor.forward_with_evidence(b, sb)
    with torch.no_grad():
        expected, _ = parent(b, parent.initial_state())
    assert torch.equal(actual, expected) and sb.option == 0 and sa.option == 1
    with pytest.raises(FrozenInstanceError):
        sa.option = 0
    cloned = sa.clone()
    assert (
        cloned.option == sa.option and cloned.recurrent[0].data_ptr() != sa.recurrent[0].data_ptr()
    )


def test_different_recurrent_parent_cannot_be_implicitly_shared_between_experts():
    parent, experts, motor, _, _ = fixture()
    changed = copy.deepcopy(experts)
    with torch.no_grad():
        changed[2].parent.head[-1].bias.add_(0.01)
    with pytest.raises(ValueError, match="EXACT"):
        CommittedNeuralMotor(
            parent, changed, motor.initiation, generation_hash="sha256:" + "a" * 64
        )


def test_option_history_requires_explicit_state_and_no_batch_sharing_or_training():
    _, _, motor, a, _ = fixture()
    with pytest.raises(ValueError):
        motor(a, (torch.zeros(2, 1, 256), torch.zeros(2, 1, 256)))
    with pytest.raises(ValueError):
        motor.initial_state(batch=2)
    with pytest.raises(ValueError):
        motor.train()
    assert all(not p.requires_grad for p in motor.parameters())
    with pytest.raises(ValueError):
        MotorOptionState(
            motor.initial_state().recurrent, motor.generation_hash, option=1, controls=0
        ).validate()


def test_contradictory_initiation_examples_cannot_be_resolved_using_case_ids():
    with pytest.raises(ValueError, match="contradictory"):
        NeuralOptionInitiation(torch.zeros(2, 288), torch.tensor([0, 1]))


def test_committed_history_cannot_cross_model_generations():
    parent, experts, motor, a, _ = fixture()
    _, state, _ = motor.forward_with_evidence(a, motor.initial_state())
    other = CommittedNeuralMotor(
        parent, experts, motor.initiation, generation_hash="sha256:" + "b" * 64
    )
    with pytest.raises(ValueError, match="generation"):
        other(a, state)


def test_unseen_initial_query_defaults_to_parent_not_an_unqualified_expert():
    _, _, motor, _, _ = fixture()
    assert motor.initiation(torch.full((1, 288), 1e6)).item() == 0
