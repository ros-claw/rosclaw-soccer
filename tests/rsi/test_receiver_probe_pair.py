import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_probe_pair import compare_receiver_probe_pair


def trajectory(n=4):
    return dict(
        qpos=np.zeros((n, 10, 79), np.float64),
        qvel=np.zeros((n, 10, 76), np.float64),
        torque=np.zeros((n, 10, 58), np.float64),
        ball_forces=np.zeros((n, 10, 2, 3), np.float64),
    )


def compare(a, b):
    return compare_receiver_probe_pair(a, b, ball_q=0, ball_v=0, receiver_q=43, receiver_v=41)


def test_contact_and_different_exposure_are_measured_not_promoted():
    base, cand = trajectory(4), trajectory(3)
    cand["ball_forces"][1, 3, 1, 0] = 5
    cand["qvel"][2, -1, 0] = 1
    result = compare(base, cand)
    assert result["common_controls"] == 3 and not result["same_exposure"]
    assert result["first_different_control"] == 1
    assert result["candidate"]["first_contact"]["first_contact_groups"] == ["left_foot"]
    assert result["candidate"]["final_relative_speed_mps"] == 1
    assert result["clean_reception_certified"] is False


def test_same_trajectory_and_modified_prefix():
    base, cand = trajectory(), trajectory()
    assert compare(base, cand)["first_different_control"] is None
    cand["qpos"][0, 0, 0] = 1
    with pytest.raises(ValueError, match="identical first"):
        compare(base, cand)


def test_nonfinite_rejected():
    base, cand = trajectory(), trajectory()
    cand["qvel"][1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite aligned"):
        compare(base, cand)


def test_half_second_after_early_contact_requires_adequate_horizon():
    base, cand = trajectory(38), trajectory(38)
    cand["ball_forces"][11, 7, 1, 0] = 5
    result = compare(base, cand)
    assert result["candidate"]["first_contact"]["half_second_observed"] is True
    assert result["candidate"]["first_contact"]["first_contact_substep"] == 117
    short = {k: v[:36] for k, v in cand.items()}
    assert compare(base, short)["candidate"]["first_contact"]["half_second_observed"] is False
