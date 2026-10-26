import numpy as np

from rosclaw_soccer.rsi.cftrack_alignment_diagnostic import direct_approach_geometry


def test_left_ball_aim_bias_violates_existing_lateral_guard_before_physics():
    result = direct_approach_geometry(
        np.array([0.0, 0.0, 0.77]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.144, 0.11]),
        np.array([5.0, -0.4, 0.11]),
    )
    assert result["heading_within_guard"] and result["forward_within_guard"]
    assert not result["lateral_within_guard"]
    assert result["lateral_offset_m"] > 0.27
    assert not result["physical_motion_evaluated"]


def test_right_ball_same_aim_geometry_can_be_attempted():
    result = direct_approach_geometry(
        np.array([0.0, 0.0, 0.77]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, -0.144, 0.11]),
        np.array([5.0, -0.4, 0.11]),
    )
    assert result["heading_within_guard"] and result["forward_within_guard"]
    assert result["lateral_within_guard"]
