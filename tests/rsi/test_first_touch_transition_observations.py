import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_action_observations import observe_contact_transition


def measured():
    return dict(
        force_n=np.zeros((300, 1, 6)),
        pelvis_z_per_substep_m=np.full((300, 1, 10), 0.7),
        ball_position_after_step_m=np.zeros((300, 1, 3)),
        navigation_command=np.zeros((300, 1, 3)),
    )


def test_delayed_knee_contact_is_distinct_from_falling():
    trace = measured()
    trace["force_n"][72, 0, 0] = 10.0
    trace["force_n"][206, 0, 4] = 20.0
    trace["navigation_command"][73:, 0, 0] = 1.4
    row = observe_contact_transition(trace)
    assert row["first_nonfoot_contact_bodies"] == ["left_knee_link"]
    assert row["frames_foot_to_nonfoot"] == 134
    assert row["foot_preceded_nonfoot"] is True
    assert row["first_low_pelvis_frame"] is None
    assert row["contact_transition_navigation_delta_m_s"] == [1.4, 0.0, 0.0]
    assert row["causation_proven"] is False
    assert row["new_physics_executions"] == 0


@pytest.mark.parametrize("frame", [None, 299])
def test_missing_postcontact_command_is_not_filled_with_zero(frame):
    trace = measured()
    if frame is not None:
        trace["force_n"][frame, 0, 1] = 10.0
    assert observe_contact_transition(trace)["contact_transition_navigation_delta_m_s"] is None


def test_nonfoot_before_foot_is_an_acquisition_failure_not_followthrough():
    trace = measured()
    trace["force_n"][20, 0, 2] = 5.0
    trace["force_n"][40, 0, 0] = 5.0
    row = observe_contact_transition(trace)
    assert row["first_any_contact_frame"] == 20
    assert row["foot_preceded_nonfoot"] is False
    assert row["frames_foot_to_nonfoot"] == -20


def test_fall_is_measured_independently_from_ball_body_contact():
    trace = measured()
    trace["pelvis_z_per_substep_m"][10, 0, 2] = 0.6
    row = observe_contact_transition(trace)
    assert row["first_low_pelvis_frame"] == 10
    assert row["first_nonfoot_contact_frame"] is None


def test_invalid_navigation_trace_rejected():
    trace = measured()
    trace["navigation_command"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        observe_contact_transition(trace)
