import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_staged_lateral_reference import lateral_stage_waypoint


@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_staged_waypoint_enters_existing_geometric_domain_without_moving_input(side):
    base = np.array([0.0, 0.0, 0.77])
    ball = np.array([1.0, side * 0.1728, 0.11])
    goal = np.array([5.0, 0.0, 0.11])
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    initial = base.copy()
    waypoint, info = lateral_stage_waypoint(base, quaternion, ball, goal)
    assert abs(info["initial_geometry"]["lateral_offset_m"]) > 0.2
    assert abs(info["planned_waypoint_geometry"]["lateral_offset_m"]) <= 0.15 + 1e-9
    assert abs(info["planned_lateral_displacement_m"]) <= 0.15
    assert info["waypoint_is_reference_not_measured_body"]
    np.testing.assert_array_equal(base, initial)
    assert waypoint[1] * side > 0


@pytest.mark.parametrize("ball", [np.array([1.0, 0.0, 0.11]), np.array([1.0, 0.5, 0.11])])
def test_stage_is_only_available_for_bounded_direct_guard_failures(ball):
    with pytest.raises(ValueError):
        lateral_stage_waypoint(
            np.array([0.0, 0.0, 0.77]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            ball,
            np.array([5.0, 0.0, 0.11]),
        )
