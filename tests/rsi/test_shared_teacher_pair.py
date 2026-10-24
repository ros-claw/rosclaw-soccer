import copy

import numpy as np
import pytest
import torch
from test_shared_native_neural_player import players

from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer
from rosclaw_soccer.rsi.shared_teacher_pair import SharedTeacherPair


def pair():
    w, ps = players()
    student = ps["a/"]
    actor = copy.deepcopy(student.actor)
    with torch.no_grad():
        list(actor.plastic.parameters())[2].add_(0.001)
        list(actor.plastic.parameters())[-1].add_(0.001)
    teacher = SharedNativeNeuralPlayer(
        w,
        "a/",
        actor,
        clip=student.clip,
        torso_index=student.torso_index,
        build_observation=student.build_observation,
        goal=student.goal,
        offset=student.offset,
        scale=student.scale,
        order=student.order,
    )
    return w, SharedTeacherPair(student, teacher)


def test_actual_student_action_feedback_is_used_by_teacher_without_h_borrowing():
    w, p = pair()
    target, s, _, t = p.propose()
    assert not np.array_equal(s["NN_action"], t["NN_action"])
    np.testing.assert_array_equal(s["NN_obs"], t["NN_obs"])
    assert p.student.state[0].data_ptr() != p.teacher.state[0].data_ptr()
    with pytest.raises(ValueError):
        p.propose()
    w.advance({"a/": target, "b/": np.zeros(29)})
    p.commit(s["NN_action"])
    np.testing.assert_array_equal(p.student.last_action, s["NN_action"])
    np.testing.assert_array_equal(p.teacher.last_action, s["NN_action"])
    np.testing.assert_array_equal(p.student.state[0].numpy(), s["NN_h_after"])
    np.testing.assert_array_equal(p.teacher.state[0].numpy(), t["NN_h_after"])
    p.begin_standing()
    target, s, _, _ = p.propose()
    w.advance({"a/": target, "b/": np.zeros(29)})
    p.commit(s["NN_action"])
    assert w.native_steps == 20


def test_expert_takeover_feedback_also_updates_own_student_history():
    w, p = pair()
    _, s, target, t = p.propose()
    w.advance({"a/": target, "b/": np.zeros(29)})
    p.commit(t["NN_action"])
    np.testing.assert_array_equal(p.student.last_action, t["NN_action"])
    np.testing.assert_array_equal(p.student.state[0].numpy(), s["NN_h_after"])
    assert not np.array_equal(p.student.state[0].numpy(), p.teacher.state[0].numpy())


def test_unexecuted_feedback_faults_pair_and_cannot_implicitly_retry():
    w, p = pair()
    target, s, _, _ = p.propose()
    w.advance({"a/": target, "b/": np.zeros(29)})
    with pytest.raises(ValueError):
        p.commit(s["NN_action"] + np.float32(0.1))
    assert p.faulted
    with pytest.raises(ValueError):
        p.propose()


def test_reference_and_body_mismatch_rejected_before_inference():
    _, p = pair()
    p.teacher.goal[0] += 0.1
    with pytest.raises(ValueError):
        SharedTeacherPair(p.student, p.teacher)
    p.teacher.goal = p.student.goal.copy()
    with torch.no_grad():
        next(p.teacher.actor.parent.parameters()).add_(0.001)
    with pytest.raises(ValueError, match="reference memory"):
        SharedTeacherPair(p.student, p.teacher)


def option_pair():
    from test_committed_neural_motor import fixture

    w, ps = players()
    old = ps["a/"]
    parent, _, actor, _, _ = fixture()
    args = dict(
        clip=old.clip,
        torso_index=old.torso_index,
        build_observation=old.build_observation,
        goal=old.goal,
        offset=old.offset,
        scale=old.scale,
        order=old.order,
    )
    return w, SharedTeacherPair(
        SharedNativeNeuralPlayer(w, "a/", actor, **args),
        SharedNativeNeuralPlayer(w, "a/", parent, **args),
    )


def test_committed_option_student_preserves_memory_with_actual_parent_teacher_feedback():
    w, pair = option_pair()
    for control in range(2):
        target, student, _, teacher = pair.propose()
        assert not np.array_equal(student["NN_action"], teacher["NN_action"])
        assert student["NN_option_after"] == 1
        w.advance({"a/": target, "b/": np.zeros(29)})
        pair.commit(student["NN_action"])
        assert pair.student.state.controls == control + 1
        assert pair.student.state.recurrent[0].data_ptr() != pair.teacher.state[0].data_ptr()
        np.testing.assert_array_equal(pair.teacher.last_action, student["NN_action"])
        if control == 0:
            pair.begin_standing()


def test_option_history_cannot_alias_teacher_recurrent_storage():
    w, pair = option_pair()
    pair.teacher.state = pair.student.state.recurrent
    with pytest.raises(ValueError, match="storage"):
        SharedTeacherPair(pair.student, pair.teacher)
    assert w.native_steps == 0
