"""Synthetic state/command diagnostics, never physical success certificates."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.native_approach_diagnostics import diagnose_native_approach


def trace(y=-0.1):
    root = np.zeros((300, 1, 7))
    root[:, 0, 0] = np.linspace(0, 4, 300)
    root[:, 0, 2] = 0.7
    root[:, 0, 6] = 1
    ball = np.zeros((300, 1, 3))
    ball[:, 0] = (2, y, 0.11)
    command = np.zeros((300, 1, 3))
    command[:, 0, 0] = 1.4
    command[:, 0, 1] = np.where((y < 0) & (2 - root[:, 0, 0] > 0.95), 1.2 * y, 0)
    return dict(
        root_pose_xyzw_m=root,
        ball_position_before_step_m=ball,
        foot_geometry_position_before_step_m=np.zeros((300, 1, 4, 3)),
        navigation_command=command,
        force_n=np.zeros((300, 1, 6)),
        motor_delta_rad=np.zeros((300, 1, 12)),
    )


def test_no_contact_can_coexist_with_overtaking_and_zero_close_tracking():
    data = trace()
    original = {key: value.copy() for key, value in data.items()}
    result = diagnose_native_approach(data)
    assert result["root_overtook_ball_before_contact"]
    assert result["first_contact_frame"] is None
    assert result["precontact_lateral_command_active_frames"] > 0
    assert result["precontact_close_longitudinal_frames"] > 0
    assert (
        result["precontact_close_longitudinal_frames"]
        == result["precontact_close_with_zero_lateral_command_frames"]
    )
    assert not result["causal_failure_reason_proven"]
    assert not result["geometry_surface_distance_checked"]
    for key in data:
        np.testing.assert_array_equal(data[key], original[key])


@pytest.mark.parametrize("y", [0, 0.1])
def test_nonnegative_initial_y_disables_legacy_tracking(y):
    result = diagnose_native_approach(trace(y))
    assert result["initial_y_disables_legacy_lateral_tracking"]
    assert result["precontact_lateral_command_active_frames"] == 0


def test_first_contact_bounds_precontact_geometry_search():
    data = trace()
    data["force_n"][50, 0, 0] = 1.01
    data["navigation_command"][51:, 0, 1] = 0
    data["foot_geometry_position_before_step_m"][49, 0, 0] = (2, -0.1, 0.11)
    result = diagnose_native_approach(data)
    assert result["first_contact_frame"] == 50
    assert result["nearest_precontact_foot_centre_frame"] == 49
    assert result["nearest_precontact_foot_centre_distance_m"] == 0
    assert not result["root_overtook_ball_before_contact"]


@pytest.mark.parametrize("key", list(trace()))
def test_reject_incomplete_frames_and_missing_arrays(key):
    data = trace()
    data[key] = data[key][:-1]
    with pytest.raises(ValueError):
        diagnose_native_approach(data)
    del data[key]
    with pytest.raises(ValueError):
        diagnose_native_approach(data)


@pytest.mark.parametrize("value", [np.nan, np.inf, 1e20])
def test_reject_invalid_measurements(value):
    data = trace()
    data["root_pose_xyzw_m"][0, 0, 0] = value
    with pytest.raises(ValueError):
        diagnose_native_approach(data)


def test_force_distance_and_threshold_are_not_interchangeable():
    data = trace(0)
    data["foot_geometry_position_before_step_m"][:] = (2, 0, 0.11)
    data["force_n"][0, 0, 0] = 1.0
    assert not diagnose_native_approach(data)["recorded_contact_detected"]
    data["force_n"][0, 0, 0] = -1
    with pytest.raises(ValueError):
        diagnose_native_approach(data)


def test_changed_command_law_is_not_silently_adopted():
    data = trace()
    data["navigation_command"][0, 0, 1] += 1e-12
    with pytest.raises(ValueError, match="command law"):
        diagnose_native_approach(data)
