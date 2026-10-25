import numpy as np
import pytest

from rosclaw_soccer.rsi.experimental_approach_command import experimental_approach_command
from rosclaw_soccer.world.player_clearance import propose_clearance_velocity


def arguments():
    return dict(
        parent=np.array([1, 0, 0], np.float32),
        previous=np.zeros(3, np.float32),
        root_xy=np.zeros(2),
        ball_xy=np.array([2.0, 0.0]),
        yaw=0.0,
        neighbor_offsets=np.empty((0, 2)),
    )


def test_fast_domain_requires_new_explicit_flag_and_old_limits_remain():
    for kwargs in ({}, {"experimental_fast_navigation": True}):
        with pytest.raises(ValueError):
            propose_clearance_velocity(
                np.array([2.0, 0]), np.empty((0, 2)), maximum_speed_mps=2, **kwargs
            )
    result = propose_clearance_velocity(
        np.array([2.0, 0]),
        np.empty((0, 2)),
        maximum_speed_mps=2,
        experimental_velocity_envelope=True,
    )
    assert result.feasible and result.velocity_mps == (2, 0)


def test_far_ball_accelerates_smoothly_and_near_ball_does_not_demand_stopping():
    args = arguments()
    for _ in range(30):
        out = experimental_approach_command(**args)
        assert not out["parent_fallback"]
        assert np.max(abs(out["command"] - args["previous"])) <= 0.0800001
        args["previous"] = out["command"].copy()
    assert out["command"][0] == pytest.approx(2)
    args["ball_xy"] = np.array([0.3, 0.0])
    for _ in range(20):
        out = experimental_approach_command(**args)
        args["previous"] = out["command"].copy()
    assert out["scale"] == 1 and out["command"][0] == pytest.approx(1)


def test_boost_does_not_undo_clearance_and_rotated_body_constraints():
    args = arguments()
    args["parent"] = np.array([0.3, 0, 0], np.float32)
    args["previous"] = args["parent"].copy()
    args["neighbor_offsets"] = np.array([[0.0, 1.1]])
    args["yaw"] = np.pi / 2
    out = experimental_approach_command(**args)
    assert out["feasible"]
    # Closing speed at1.1m may not exceed .6*(1.1-.6)=.3m/s.
    assert out["command"][0] <= 0.300001
    assert np.max(abs(out["command"] - args["previous"])) <= 0.0800001


def test_infeasible_intersection_is_explicit_parent_fallback():
    args = arguments()
    args["neighbor_offsets"] = np.array([[0.1, 0]])
    out = experimental_approach_command(**args)
    assert out["parent_fallback"] and not out["feasible"]
    np.testing.assert_array_equal(out["command"], args["parent"])


def test_invalid_history_is_rejected():
    args = arguments()
    args["previous"][0] = 2.1
    with pytest.raises(ValueError):
        experimental_approach_command(**args)
