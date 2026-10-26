import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_staged_lateral_reference import (
    extend_stage_reference,
    lateral_stage_waypoint,
)


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


def test_stage_hold_repeats_reference_only_and_preserves_tail():
    prefix = {
        "ref_base_pos": np.arange(15, dtype=np.float32).reshape(5, 3),
        "ref_phase": np.array([12, 12, 12, 13, 14], dtype=np.int64),
    }
    old = {key: value.copy() for key, value in prefix.items()}
    result = extend_stage_reference(prefix, stage_controls=3, hold_controls=2)
    np.testing.assert_array_equal(
        result["ref_base_pos"][3:5], np.repeat(old["ref_base_pos"][2:3], 2, axis=0)
    )
    np.testing.assert_array_equal(result["ref_base_pos"][5:], old["ref_base_pos"][3:])
    np.testing.assert_array_equal(result["ref_phase"], [12, 12, 12, 12, 12, 13, 14])
    for key in old:
        np.testing.assert_array_equal(prefix[key], old[key])


@pytest.mark.parametrize("hold", [-1, 21, True])
def test_invalid_hold_fails_closed(hold):
    with pytest.raises(ValueError):
        extend_stage_reference({"ref": np.ones((4, 3))}, stage_controls=2, hold_controls=hold)


def test_stage_hold_rejects_incompatible_or_nonfinite_arrays():
    with pytest.raises(ValueError):
        extend_stage_reference(
            {"a": np.ones((4, 3)), "b": np.ones((5,))}, stage_controls=2, hold_controls=1
        )
    with pytest.raises(ValueError):
        extend_stage_reference({"a": np.array([[np.nan], [0]])}, stage_controls=2, hold_controls=1)
