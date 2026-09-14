import numpy as np
import pytest

from rosclaw_soccer.physics.goal_entry import audit_goal_entries
from rosclaw_soccer.world.field import G1TrainingGoalSpec


def test_actual_s1906_rolling_goal_cannot_be_credited_as_save():
    # Measured 50 Hz native-physics frames: full ball enters despite soft floor
    # penetration. Keep this regression independent of external evidence paths.
    goal = G1TrainingGoalSpec(
        plane_x_m=7.5,
        width_m=7.32,
        height_m=2.44,
        ball_radius_m=0.10981691073340778,
        ball_mass_kg=0.43,
    )
    result = audit_goal_entries(
        time_sec=np.array([9.939999999999989, 9.959999999999996]),
        ball_position_m=np.array(
            [
                [7.6047904534368, -2.9420692710808036, 0.10979241975521103],
                [7.609931494963847, -2.935807768516753, 0.10980616491369734],
            ]
        ),
        goal=goal,
    )
    assert result.goal_observed
    assert result.first_goal_time_sec == pytest.approx(9.959554237288932)
    assert result.whole_ball_plane_x_m == pytest.approx(7.609816910733408)


def audit(xs, *, ys=None, zs=None, times=None, tolerance=0.001):
    goal = G1TrainingGoalSpec()
    count = len(xs)
    return audit_goal_entries(
        time_sec=np.arange(count) * 0.02 if times is None else np.asarray(times),
        ball_position_m=np.column_stack(
            (
                xs,
                np.zeros(count) if ys is None else ys,
                np.full(count, goal.ball_radius_m - 0.00002) if zs is None else zs,
            )
        ),
        goal=goal,
        ground_contact_tolerance_m=tolerance,
    )


def test_rolling_goal_is_not_a_numerical_floor_save():
    goal = G1TrainingGoalSpec()
    result = audit([goal.plane_x_m, goal.plane_x_m + 0.3])
    assert result.goal_observed
    assert result.goal_entries == 1
    assert not audit([goal.plane_x_m, goal.plane_x_m + 0.3], tolerance=0).goal_observed


def test_center_crossing_is_not_whole_ball_entry():
    goal = G1TrainingGoalSpec()
    assert not audit([goal.plane_x_m - 0.01, goal.plane_x_m + 0.01]).goal_observed


def test_later_valid_crossing_is_not_hidden_by_first_miss():
    goal = G1TrainingGoalSpec()
    p = goal.plane_x_m
    result = audit([p - 0.2, p + 0.3, p - 0.2, p + 0.3], ys=[10, 10, 0, 0])
    assert result.complete_forward_crossings == 2
    assert result.goal_entries == 1
    assert result.first_goal_time_sec > 0.04


@pytest.mark.parametrize("height", [-0.1, 3.0])
def test_no_floor_or_crossbar_enlargement(height):
    p = G1TrainingGoalSpec().plane_x_m
    assert not audit([p, p + 0.3], zs=[height, height]).goal_observed


def test_outside_post_and_reverse_crossing_are_not_goals():
    p = G1TrainingGoalSpec().plane_x_m
    assert not audit([p, p + 0.3], ys=[10, 10]).goal_observed
    result = audit([p, p + 0.3, p], ys=[10, 10, 0])
    assert result.complete_forward_crossings == 1
    assert not result.goal_observed


@pytest.mark.parametrize("times", [[0, 0], [0, 0.021], [0, float("nan")], [-0.01, 0.01]])
def test_bad_clock_rejected(times):
    p = G1TrainingGoalSpec().plane_x_m
    with pytest.raises(ValueError):
        audit([p, p + 0.3], times=times)


@pytest.mark.parametrize("tolerance", [-0.001, 0.002, float("nan"), True])
def test_tolerance_validated_even_without_crossing(tolerance):
    with pytest.raises(ValueError):
        audit([0, 0.01], tolerance=tolerance)


def test_nonfinite_position_and_missing_prefix_rejected():
    with pytest.raises(ValueError):
        audit([0, float("inf")])
    with pytest.raises(ValueError):
        audit([100, 101])
    with pytest.raises(ValueError, match="overflow"):
        audit([-1e308, 1e308])


def test_audit_does_not_modify_inputs():
    goal = G1TrainingGoalSpec()
    positions = np.array([[goal.plane_x_m, 0, 0.2], [goal.plane_x_m + 0.3, 0, 0.2]])
    times = np.array([0.0, 0.02])
    before = positions.copy()
    result = audit_goal_entries(time_sec=times, ball_position_m=positions, goal=goal)
    np.testing.assert_array_equal(positions, before)
    np.testing.assert_array_equal(times, [0, 0.02])
    assert result.goal_observed
