"""Tiny native-world adapter DAgger bridge checks, not football qualification."""

import copy

import numpy as np
import pytest
import torch
from test_consolidated_adapter_native_bridge import make_adapter_player
from test_shared_native_neural_player import players

from rosclaw_soccer.rsi.consolidated_motor_adapter import ConsolidatedMotorAdapter
from rosclaw_soccer.rsi.same_state_motor_pair import SameStateMotorPair, paired_reference
from rosclaw_soccer.rsi.shared_teacher_pair import SharedTeacherPair


def models():
    world, original = players()
    student = copy.deepcopy(original["a/"].actor)
    teacher = copy.deepcopy(student)
    return world, original["a/"], student, teacher


@pytest.mark.parametrize(
    "adapt_student,adapt_teacher", [(True, False), (False, True), (True, True)]
)
def test_adapter_pair_consumes_actual_actions_and_keeps_private_state_through_standing(
    adapt_student, adapt_teacher
):
    world, original, student, teacher = models()
    if adapt_student:
        student = ConsolidatedMotorAdapter(student, head_bound_rad=0.2)
    if adapt_teacher:
        teacher = ConsolidatedMotorAdapter(teacher, head_bound_rad=0.2)
        with torch.no_grad():
            teacher.head[-1].bias.add_(0.01)
    else:
        with torch.no_grad():
            teacher.plastic.head.bias.add_(0.01)
    student.requires_grad_(False).eval()
    teacher.requires_grad_(False).eval()
    p = SharedTeacherPair(
        make_adapter_player(world, original, student),
        make_adapter_player(world, original, teacher),
    )
    pure = SameStateMotorPair(student, teacher)
    target, s, expert, t = p.propose()
    np.testing.assert_array_equal(s["NN_obs"], t["NN_obs"])
    assert not np.array_equal(s["NN_action"], t["NN_action"])
    own, advice, state, evidence = pure.observe(
        torch.from_numpy(s["NN_obs"]).reshape(1, 1, 160), pure.initial_state()
    )
    np.testing.assert_array_equal(own[0, 0].numpy(), s["NN_action"])
    np.testing.assert_array_equal(advice[0, 0].numpy(), t["NN_action"])
    assert evidence["teacher_advice_physically_qualified"] is False
    assert state[0][0].data_ptr() != state[1][0].data_ptr()
    # Execute the student, then let the expert take over the SAME actual body.
    world.advance({"a/": target, "b/": np.zeros(29)})
    p.commit(s["NN_action"])
    p.begin_standing()
    _, s, expert, t = p.propose()
    world.advance({"a/": expert, "b/": np.zeros(29)})
    p.commit(t["NN_action"])
    np.testing.assert_array_equal(p.student.last_action, t["NN_action"])
    np.testing.assert_array_equal(p.teacher.last_action, t["NN_action"])
    np.testing.assert_array_equal(p.student.state[0].numpy(), s["NN_h_after"])
    np.testing.assert_array_equal(p.teacher.state[0].numpy(), t["NN_h_after"])
    assert world.native_steps == 20


@pytest.mark.parametrize("alias", ["head", "memory"])
def test_adapter_pair_rejects_shared_nonreference_parameter_storage(alias):
    _, _, a, b = models()
    student = ConsolidatedMotorAdapter(a).requires_grad_(False)
    teacher = ConsolidatedMotorAdapter(b).requires_grad_(False)
    if alias == "head":
        teacher.head = student.head
    else:
        teacher.memory.plastic = student.memory.plastic
    with pytest.raises(ValueError, match="alias"):
        SameStateMotorPair(student, teacher)


def test_adapter_reference_mismatch_is_rejected_without_physics_or_hidden_borrowing():
    world, original, a, b = models()
    student = ConsolidatedMotorAdapter(a).requires_grad_(False)
    teacher = b.requires_grad_(False)
    with torch.no_grad():
        next(paired_reference(student).parameters()).add_(0.01)
    with pytest.raises(ValueError, match="reference memory"):
        SharedTeacherPair(
            make_adapter_player(world, original, student),
            make_adapter_player(world, original, teacher),
        )
    assert world.native_steps == 0


def test_unadmitted_subclass_and_trainable_adapter_do_not_get_a_teacher_pair():
    _, _, a, b = models()
    teacher = b.requires_grad_(False)
    with pytest.raises(ValueError):
        SameStateMotorPair(ConsolidatedMotorAdapter(a), teacher)

    class ArbitraryWrapper(ConsolidatedMotorAdapter):
        pass

    subclass = ArbitraryWrapper(a).requires_grad_(False)
    with pytest.raises(ValueError):
        SameStateMotorPair(subclass, teacher)
    with pytest.raises(ValueError):
        paired_reference(subclass)
