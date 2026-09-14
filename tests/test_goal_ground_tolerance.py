import pytest

from rosclaw_soccer.world.field import (
    G1TrainingGoalSpec,
    g1_ball_inside_goal_mouth,
)


def test_strict_default_unchanged_and_floor_tolerance_explicit():
    goal = G1TrainingGoalSpec(plane_x_m=12, width_m=3, height_m=2)
    assert not g1_ball_inside_goal_mouth(goal, ball_y_m=0, ball_z_m=0.11498268505638988)
    assert g1_ball_inside_goal_mouth(
        goal, ball_y_m=0, ball_z_m=0.11498268505638988, ground_contact_tolerance_m=0.001
    )
    assert not g1_ball_inside_goal_mouth(
        goal, ball_y_m=0, ball_z_m=0.1139, ground_contact_tolerance_m=0.001
    )


@pytest.mark.parametrize("y,z", [(1.386, 0.115), (0, 1.886), (0, -0.001)])
def test_tolerance_cannot_expand_sides_crossbar_or_allow_buried_ball(y, z):
    assert not g1_ball_inside_goal_mouth(
        G1TrainingGoalSpec(width_m=3, height_m=2),
        ball_y_m=y,
        ball_z_m=z,
        ground_contact_tolerance_m=0.001,
    )


@pytest.mark.parametrize("value", [True, None, "0", -0.001, 0.001001, float("nan"), float("inf")])
def test_invalid_tolerance_rejected(value):
    with pytest.raises(ValueError, match="tolerance"):
        g1_ball_inside_goal_mouth(
            G1TrainingGoalSpec(),
            ball_y_m=0,
            ball_z_m=0.115,
            ground_contact_tolerance_m=value,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_position_never_passes_with_tolerance(value):
    goal = G1TrainingGoalSpec()
    assert not g1_ball_inside_goal_mouth(
        goal, ball_y_m=value, ball_z_m=0.115, ground_contact_tolerance_m=0.001
    )
    assert not g1_ball_inside_goal_mouth(
        goal, ball_y_m=0, ball_z_m=value, ground_contact_tolerance_m=0.001
    )
