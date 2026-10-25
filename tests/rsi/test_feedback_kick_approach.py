import numpy as np
import pytest

from rosclaw_soccer.rsi.feedback_kick_approach import velocity_matched_approach


def request(gap, vx):
    return dict(
        root_position=np.array([0.0, 0.0, 0.793]),
        ball_position=np.array([gap, 0.0, 0.11]),
        ball_velocity=np.array([vx, 0.0, 0.0]),
        cruise_speed=1.5,
        entry_gap=0.9,
        lateral_offset=0.0,
    )


def test_measured_motion_changes_command_at_same_geometry():
    assert velocity_matched_approach(**request(2.5, -0.7))[0] == 1.5
    assert velocity_matched_approach(**request(1.0, -0.7))[0] == 0
    assert velocity_matched_approach(**request(1.0, 0.7))[0] == pytest.approx(1.15)
    assert velocity_matched_approach(**request(1.0, 0.0))[0] == pytest.approx(0.45)
    assert velocity_matched_approach(**request(0.9, 0.0))[0] == pytest.approx(0.3)


def test_feedback_has_no_timer_or_state_write_and_bounded_lateral():
    args = request(1.2, 0.0)
    args["ball_velocity"][1] = 2
    before = args["ball_velocity"].copy()
    assert velocity_matched_approach(**args)[1] == 0.4
    np.testing.assert_array_equal(before, args["ball_velocity"])


@pytest.mark.parametrize("key", ["ball_velocity", "root_position", "ball_position"])
def test_nonfinite_input_never_becomes_a_command(key):
    args = request(1.2, 0.0)
    args[key][0] = np.nan
    with pytest.raises(ValueError):
        velocity_matched_approach(**args)
