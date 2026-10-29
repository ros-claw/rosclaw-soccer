import numpy as np
import pytest

from rosclaw_soccer.rsi.precontact_knee_intervention import (
    KNEES,
    KneeInterventionState,
    research_knee_target,
)


def step(state, gaps, *, offset=0.04, foot=False):
    target = np.zeros(29, dtype=np.float64)
    result = research_knee_target(
        state,
        control=state.signal.next_control,
        target=target,
        knee_ball_surface_gaps_m=gaps,
        actual_foot_contact=foot,
        offset_rad=offset,
    )
    assert not target.any()
    return result


def test_trigger_once_and_exact_five_control_lease():
    state, target = step(KneeInterventionState(), (0.09, 0.2))
    assert not target.any()
    state, target = step(state, (0.055, 0.2))
    assert not target.any()
    state, target = step(state, (0.045, 0.2))
    assert target[KNEES[0]] == 0.04 and np.count_nonzero(target) == 1
    for _ in range(4):
        state, target = step(state, (0.04, 0.2), foot=True)
        assert target[KNEES[0]] == 0.04
    state, target = step(state, (0.03, 0.2))
    assert not target.any() and state.fired and state.remaining_controls == 0


def test_zero_arm_records_trigger_without_changing_target():
    state, _ = step(KneeInterventionState(), (0.09, 0.09), offset=0.0)
    state, _ = step(state, (0.055, 0.09), offset=0.0)
    state, target = step(state, (0.045, 0.09), offset=0.0)
    assert state.fired and not target.any()


@pytest.mark.parametrize("offset", [0.01, -0.4, float("nan"), 1])
def test_offset_outside_declared_bank_rejected(offset):
    with pytest.raises(ValueError):
        step(KneeInterventionState(), (0.1, 0.1), offset=offset)
