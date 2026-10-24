import copy

import pytest
import torch

from rosclaw_soccer.rsi.same_state_motor_pair import SameStateMotorPair


def test_option_student_and_parent_advisor_cannot_share_private_history():
    from test_committed_neural_motor import fixture

    parent, _, student, a, b = fixture()
    p = SameStateMotorPair(student, parent)
    states = p.initial_state()
    for obs in (a, b, b):
        actual, advice, states, _ = p.observe(obs, states)
        assert states[0].option == 1
        assert states[0].recurrent[0].data_ptr() != states[1][0].data_ptr()
    assert not torch.equal(actual, advice)
    with pytest.raises(ValueError, match="borrow"):
        p.observe(a, (states[0], states[0].recurrent))


def pair():
    from test_dual_recurrent_motor import model

    _, student = model()
    teacher = copy.deepcopy(student)
    with torch.no_grad():
        teacher.plastic.head.bias.add_(0.1)
    return SameStateMotorPair(student.requires_grad_(False), teacher.requires_grad_(False))


def test_actual_control_is_student_and_teacher_only_observes_same_executed_history():
    p = pair()
    state, private = p.initial_state(), p.student.initial_state()
    for _ in range(8):
        observation = torch.randn(1, 1, 160)
        actual, advice, state, evidence = p.observe(observation, state)
        with torch.no_grad():
            expected, private = p.student(observation, private)
        assert torch.equal(actual, expected)
        assert not torch.equal(actual, advice)
        assert torch.equal(state[0][0], private[0])
        assert evidence["teacher_advice_is_executed"] is False
        assert evidence["teacher_advice_physically_qualified"] is False


def test_hidden_state_ownership_and_teacher_history_are_independent():
    p = pair()
    state = p.initial_state()
    original = tuple(tuple(t.clone() for t in s) for s in state)
    changed = tuple(tuple(t.clone() for t in s) for s in state)
    changed[0][0][..., 128:] += 2
    observation = torch.randn(1, 1, 160)
    actual, advice, after, _ = p.observe(observation, original)
    other, same_advice, _, _ = p.observe(observation, changed)
    assert not torch.equal(actual, other)
    assert torch.equal(advice, same_advice)
    assert all(torch.equal(t, torch.zeros_like(t)) for s in state for t in s)
    assert after[0][0].untyped_storage().data_ptr() != after[1][0].untyped_storage().data_ptr()
    with pytest.raises(ValueError, match="borrow"):
        p.observe(observation, (state[0], state[0]))


def test_pair_cannot_use_same_network_or_trainable_network_as_qualified_teacher():
    p = pair()
    with pytest.raises(ValueError):
        SameStateMotorPair(p.student, p.student)
    p.teacher.plastic.requires_grad_(True)
    with pytest.raises(ValueError):
        SameStateMotorPair(p.student, p.teacher)


def test_refinement_and_whole_acquired_parent_keep_private_same_state_histories():
    from test_refinement_motor_adapter import parent_model

    from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter

    parent = parent_model().eval().requires_grad_(False)
    refined = RefinementMotorAdapter(parent).eval().requires_grad_(False)
    p = SameStateMotorPair(refined, parent)
    state = p.initial_state()
    for _ in range(3):
        actual, advice, state, _ = p.observe(torch.randn(1, 1, 160), state)
        assert torch.equal(actual, advice)
        assert state[0][0].untyped_storage().data_ptr() != state[1][0].untyped_storage().data_ptr()
