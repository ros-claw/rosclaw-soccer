from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_paid_foot_taskspace import synthetic_episode

from rosclaw_soccer.rsi.paid_causal_context import CausalMotorContext
from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent
from rosclaw_soccer.rsi.paid_own_state_teacher import OwnStateTeacherHandoff, handoff_controls
from rosclaw_soccer.rsi.paid_trainable_actor import PaidTrainableActor


def fixture():
    ep = synthetic_episode()
    ep.clip = SimpleNamespace(length=289)
    # Synthetic contract fixture, not a trained or admitted reference actor.
    base = PaidTrainableActor.__new__(PaidTrainableActor)
    torch.nn.Module.__init__(base)
    base.actor = torch.nn.Linear(128, 29)
    with torch.no_grad():
        base.actor.weight.zero_()
        base.actor.weight[:, 0] = 1
        base.actor.bias.zero_()
    base.eval().requires_grad_(False)
    intent = FootTaskSpaceIntent(ep.env.model, foot="left", center=120, latent=np.zeros((2, 6)))
    teacher = OwnStateTeacherHandoff(ep, base, intent=intent, switch_control=109)
    return ep, teacher


def context(ep, value=0.1):
    before = np.full((2, 1, 128), 99, np.float32)
    after = np.full((2, 1, 128), value, np.float32)
    return CausalMotorContext(
        id(ep), ep.frame, np.zeros(160, np.float32), before, before, after, after
    )


def test_teacher_uses_current_own_after_hidden_and_explicit_host_clock():
    ep, teacher = fixture()
    student = np.full(29, 0.11, np.float32)
    action, p = teacher(ep, student, context(ep))
    assert np.array_equal(action, student) and not p["teacher_applied"]
    assert np.array_equal(p["R2_nominal_action"], np.full(29, 0.1, np.float32))
    assert np.array_equal(p["teacher_action"], p["R2_nominal_action"])
    ep.frame = 109
    action, p = teacher(ep, student, context(ep, 0.2))
    assert p["teacher_applied"] and np.array_equal(action, np.full(29, 0.2, np.float32))
    assert teacher.student_rows == teacher.teacher_rows == 1
    assert teacher.r2_head_queries == teacher.ik_queries == 2
    assert np.array_equal(student, np.full(29, 0.11, np.float32))


def test_teacher_cannot_borrow_state_or_expand_student_authority():
    ep, teacher = fixture()
    c = context(ep)
    ep.frame += 1
    with pytest.raises(ValueError):
        teacher(ep, np.zeros(29, np.float32), c)
    with pytest.raises(ValueError):
        teacher(ep, np.ones(29, np.float32), context(ep))
    other, _ = fixture()
    with pytest.raises(ValueError):
        teacher(other, np.zeros(29, np.float32), context(other))


def test_boundaries_are_predeclared_from_teacher_phase_not_student_outcome():
    assert handoff_controls(120, 289) == dict(
        teacher=0, phase_start=78, middle=96, late=112, student=289
    )
    for centre, horizon in ((True, 289), (20, 289), (120, 150), (120, 301)):
        with pytest.raises(ValueError):
            handoff_controls(centre, horizon)


def test_dirty_handoff_runner_rejects_before_reading_weights_or_physics(tmp_path, monkeypatch):
    import sys

    from scripts import rsi_probe_paid_own_state_teacher as runner

    args = ["run", "--expected-head", "frozen", "--output", str(tmp_path / "out")]
    for name in ("teacher", "teacher-audit", "student", "student-exam", "student-audit"):
        args += ["--" + name + "-root", str(tmp_path / name), "--" + name + "-hash", "hash"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="frozen"):
        runner.main()
    assert not (tmp_path / "out").exists()


def test_dirty_handoff_audit_rejects_before_wait_or_output(tmp_path, monkeypatch):
    import sys

    from scripts import rsi_audit_paid_own_state_teacher as runner

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--root",
            str(tmp_path / "pilot"),
            "--producer-head",
            "producer",
            "--expected-head",
            "frozen",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: b"dirty")
    with pytest.raises(ValueError, match="frozen"):
        runner.main()
    assert not (tmp_path / "out").exists()
