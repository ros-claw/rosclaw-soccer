"""Orchestration mocks only: real history/native qualification remains mandatory."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.rsi_warm_student_intervention import warm_student30


def test_actual_student_action_not_teacher_advice_and_no_clock_reset(monkeypatch):
    from rosclaw_soccer.rsi import ball_entry_teacher as entry
    from rosclaw_soccer.rsi import full_entry_forecast as forecast
    from scripts import rsi_motion_student_rollout as student
    from scripts import rsi_motion_teacher_tap as taps
    from scripts import rsi_motor_student_rollout as motor

    q, v = np.zeros(43), np.zeros(41)
    feature, token, hidden, muscle = (np.zeros(n, np.float32) for n in (108, 64, 128, 930))
    action = np.zeros(29, np.float32)
    expected = {k: np.zeros((300, 1)) for k in motor.PHYSICAL_FIELDS}
    expected.update(
        canonical_qpos=np.zeros((300, 43)),
        canonical_qvel=np.zeros((300, 41)),
        student_features=np.zeros((300, 108), np.float32),
        student_motion_token=np.zeros((300, 64), np.float32),
        hidden_after=np.zeros((300, 128), np.float32),
        muscle_proprioception=np.zeros((300, 930), np.float32),
        consumed_action_isaac=np.zeros((300, 29), np.float32),
        value_estimate=np.zeros(300, np.float32),
    )
    monkeypatch.setattr(student, "validate_motion_recording", lambda *a, **kw: None)
    monkeypatch.setattr(student, "disable_teacher_networks", lambda c: None)
    monkeypatch.setattr(student, "proprioception", lambda c: muscle.copy())
    model = NS(calls=0)

    def prediction(m, *args):
        m.calls += 1
        return feature.copy(), token.copy(), np.float32(0), None, hidden.copy()

    monkeypatch.setattr(student, "prediction", prediction)
    monkeypatch.setattr(motor, "action_target", lambda a: np.zeros(29))
    monkeypatch.setattr(entry, "ball_entry_decision", lambda *a, **kw: NS(command=(0.0, 0.0, 0.0)))
    monkeypatch.setattr(forecast, "_observation", lambda n, b, f, c: NS(frame=f))
    body = NS(
        next_frame=0,
        integrated_steps=0,
        attempted_steps=0,
        model=NS(nbody=46),
        kp=np.ones(29),
        kd=np.ones(29),
        canonical=lambda: (q.copy(), v.copy()),
    )

    def step(frame, target):
        assert frame == body.next_frame
        body.next_frame += 1
        body.integrated_steps += 10
        body.attempted_steps += 10
        return {k: expected[k][frame : frame + 1].copy() for k in motor.PHYSICAL_FIELDS}

    body.step = step
    codec = NS(_next_frame=0, _backend=object(), decode_motion=lambda t: action.copy())
    codec.observe = lambda *args, **kw: None

    def consume(a, *, foundation_prefix):
        assert foundation_prefix == (codec._next_frame < 20)
        assert not a.any()
        codec._next_frame += 1

    codec.commit_consumed = consume
    teacher = NS(
        _next_frame=0,
        _origin_frame=0,
        config=NS(pose_reference=None, latent_schedule=None),
        backend=NS(_encoder=object(), _decoder=object()),
    )

    class Tap:
        def __init__(self, backend):
            self.encoder = NS(calls=0, session=backend._encoder)
            self.decoder = NS(calls=0, session=backend._decoder)

        def label(self, actual):
            assert actual.tobytes() == muscle.tobytes()
            self.encoder.calls += 1
            self.decoder.calls += 1
            return dict(teacher_advice_action=advice(NS(frame=teacher._next_frame)))

    def advice(obs):
        return np.full(29, float(obs.frame >= 20), dtype=np.float32)

    teacher.prepare_shadow_action = advice

    def commit(a):
        assert not a.any()  # Advice after20 is all ones, but must NEVER execute.
        teacher._next_frame += 1
        return NS(target_rad=np.zeros(29), kp=body.kp.copy(), kd=body.kd.copy())

    teacher.commit_batched_action = commit
    monkeypatch.setattr(taps, "MotionTeacherTap", Tap)
    actual, labels = warm_student30(model, codec, teacher, body, None, expected, progress={})
    assert body.next_frame == teacher._next_frame == codec._next_frame == model.calls == 30
    assert body.integrated_steps == body.attempted_steps == 300
    assert not actual["consumed_action_isaac"].any()
    assert labels["teacher_advice_action"][20:].all()
    with pytest.raises(ValueError, match="fresh independent"):
        warm_student30(model, codec, teacher, body, None, expected, progress={})
