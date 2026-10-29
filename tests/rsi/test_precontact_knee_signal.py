import math

import pytest

from rosclaw_soccer.rsi.precontact_knee_signal import KneeSignalState, precontact_knee_signal


def step(state, gaps, foot=False):
    return precontact_knee_signal(
        state, control=state.next_control, knee_ball_surface_gaps_m=gaps, actual_foot_contact=foot
    )


def test_two_same_side_transitions_required_and_no_automatic_latch():
    state = KneeSignalState()
    state, signal = step(state, (0.09, 0.09))
    assert signal == (False, False)
    state, signal = step(state, (0.055, 0.09))
    assert signal == (False, False)
    state, signal = step(state, (0.045, 0.09))
    assert signal == (True, False)
    state, signal = step(state, (0.05, 0.09))
    assert signal == (False, False)


def test_cross_side_progress_cannot_form_persistence():
    state, _ = step(KneeSignalState(), (0.09, 0.09))
    state, _ = step(state, (0.05, 0.09))
    state, signal = step(state, (0.07, 0.05))
    assert signal == (False, False)


def test_contact_is_latched_even_if_next_sample_releases():
    state, _ = step(KneeSignalState(), (0.09, 0.09))
    state, _ = step(state, (0.05, 0.09), foot=True)
    state, signal = step(state, (0.04, 0.09))
    assert signal == (False, False)
    assert state.foot_contact_seen


@pytest.mark.parametrize("bad", [(math.nan, 0.1), (math.inf, 0.1), (True, 0.1), [0.1, 0.2]])
def test_invalid_input_is_rejected_without_mutating_state(bad):
    state = KneeSignalState()
    with pytest.raises(ValueError):
        precontact_knee_signal(
            state, control=0, knee_ball_surface_gaps_m=bad, actual_foot_contact=False
        )
    assert state == KneeSignalState()


def test_invalid_control_and_non_boolean_contact_rejected():
    with pytest.raises(ValueError):
        precontact_knee_signal(
            KneeSignalState(),
            control=1,
            knee_ball_surface_gaps_m=(0.1, 0.1),
            actual_foot_contact=False,
        )
    with pytest.raises(ValueError):
        precontact_knee_signal(
            KneeSignalState(), control=0, knee_ball_surface_gaps_m=(0.1, 0.1), actual_foot_contact=1
        )
