from types import SimpleNamespace

import numpy as np
import pytest
from test_shared_native_torque_layer import world

from rosclaw_soccer.rsi.shared_native_receive_teacher import SharedNativeReceiveTeacher


def test_live_teacher_keeps_one_shared_clock_and_effort_limits():
    w = world()
    teacher = SharedNativeReceiveTeacher(w, "b/", horizon_controls=2)
    targets = {p: np.zeros(29) for p in w.layouts}
    output = w.advance(targets, torque_layers={"b/": teacher})
    assert teacher.commands == teacher.applied == 10
    assert w.native_steps == 10 and w.control_steps == 1
    assert output["torque"].shape == (10, 58)
    assert np.all(np.abs(output["torque"][:, w.layouts["b/"].ai]) <= w.parameters["b/"].effort)
    np.testing.assert_array_equal(
        output["torque"][:, w.layouts["a/"].ai],
        output["parent_torque"][:, w.layouts["a/"].ai],
    )


def test_teacher_can_join_only_at_actual_boundary_and_cannot_be_skipped():
    w = world()
    targets = {p: np.zeros(29) for p in w.layouts}
    w.advance(targets)
    teacher = SharedNativeReceiveTeacher(w, "b/", horizon_controls=1)
    with pytest.raises(ValueError, match="all registered"):
        w.advance(targets)
    assert w.native_steps == 10
    w.advance(targets, torque_layers={"b/": teacher})
    with pytest.raises(ValueError, match="bound private"):
        w.advance(targets, torque_layers={"b/": teacher})


def test_teacher_does_not_accept_counterfeit_native_feedback():
    w = world()
    teacher = SharedNativeReceiveTeacher(w, "b/", horizon_controls=1)
    with pytest.raises(ValueError, match="actual applied"):
        teacher.note_applied(np.zeros(29))
    assert teacher.failed


def test_teacher_residual_reaches_actual_motor_and_never_exceeds_effort(monkeypatch):
    import rosclaw_soccer.rsi.shared_native_receive_teacher as module

    monkeypatch.setattr(
        module,
        "locomotion_contact_teacher_effect",
        lambda **kwargs: SimpleNamespace(torque_nm=np.full(29, 100.0), active=True),
    )
    w = world()
    teacher = SharedNativeReceiveTeacher(w, "b/", horizon_controls=1)
    output = w.advance({p: np.zeros(29) for p in w.layouts}, torque_layers={"b/": teacher})
    assert teacher.active_commands == 10
    np.testing.assert_array_equal(output["torque"][:, w.layouts["b/"].ai], 25.0)
    assert np.any(output["applied_torque_change"][:, w.layouts["b/"].ai])
