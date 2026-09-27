import numpy as np
import pytest

from rosclaw_soccer.sim.virtual_goal_evidence import first_virtual_goal_crossing


def test_interpolated_whole_ball_goal() -> None:
    ball = np.array([[4.9, 0.5, 0.11], [5.2, 0.7, 0.11]])
    crossing = first_virtual_goal_crossing(ball)
    assert crossing is not None
    assert crossing.frame_after == 1
    assert crossing.center_y_m == pytest.approx(0.64)
    assert crossing.whole_ball_inside


def test_rejects_miss_and_invalid_trajectory() -> None:
    assert first_virtual_goal_crossing(np.array([[4.0, 0.0, 0.11], [4.5, 0.0, 0.11]])) is None
    crossing = first_virtual_goal_crossing(np.array([[5.0, 1.2, 0.11], [5.2, 1.2, 0.11]]))
    assert crossing is not None and not crossing.whole_ball_inside
    with pytest.raises(ValueError):
        first_virtual_goal_crossing(np.array([[4.0, np.nan, 0.11], [5.2, 0.0, 0.11]]))
