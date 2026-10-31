import pytest

from rosclaw_soccer.skills.team.motor_handoff import ExclusiveMotorHandoff, MotorOwner


def step(state, frame, **events):
    values = dict(
        native_active=False, learned_claim=False, learned_release=False, learned_fault=False
    )
    values.update(events)
    return state.advance(frame=frame, **values)


def test_defer_then_claim_hold_release_and_native_resume():
    s = ExclusiveMotorHandoff()
    assert step(s, 0, native_active=True, learned_claim=True).claim_deferred
    assert step(s, 1, learned_claim=True).owner is MotorOwner.LEARNED
    assert step(s, 2).owner is MotorOwner.LEARNED
    assert step(s, 3, learned_release=True).owner is MotorOwner.NONE
    assert step(s, 4, native_active=True).owner is MotorOwner.NATIVE


def test_overlap_faults_and_stays_faulted():
    s = ExclusiveMotorHandoff()
    step(s, 0, learned_claim=True)
    with pytest.raises(ValueError, match="overlaps"):
        step(s, 1, native_active=True)
    with pytest.raises(ValueError, match="remains faulted"):
        step(s, 1)


def test_fault_is_terminal_and_never_grants_native_fallback():
    s = ExclusiveMotorHandoff()
    assert step(s, 0, native_active=True, learned_fault=True).owner is MotorOwner.FAULT
    with pytest.raises(ValueError, match="remains faulted"):
        step(s, 1, native_active=True)


def test_invalid_events_do_not_consume_a_frame():
    s = ExclusiveMotorHandoff()
    with pytest.raises(ValueError):
        step(s, 0, learned_release=True)
    with pytest.raises(ValueError):
        step(s, 1)
    with pytest.raises(ValueError):
        step(s, 0, learned_claim=1)
    assert step(s, 0).owner is MotorOwner.NONE
