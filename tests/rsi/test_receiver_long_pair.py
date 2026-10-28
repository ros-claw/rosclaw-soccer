import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_long_pair import compare_long_pair


def sample(controls):
    return {
        "qpos": np.zeros((controls, 10, 79), np.float64),
        "qvel": np.zeros((controls, 10, 76), np.float64),
        "torque": np.zeros((controls, 10, 58), np.float64),
        "ball_forces": np.zeros((controls, 10, 2, 3), np.float64),
        "executed_receiver_action": np.zeros((2, 29), np.float32),
        "receiver_observation": np.zeros((2, 98), np.float32),
    }


LAYOUT = dict(ball_q=0, ball_v=0, receiver_q=7, receiver_v=6)


def test_common_horizon_prevents_censored_followup_being_claimed():
    b, c = sample(100), sample(20)
    b["ball_forces"][1, 0, 1, 0] = c["ball_forces"][1, 0, 1, 0] = 2
    result = compare_long_pair(b, c, layout=LAYOUT)
    assert result["common_controls"] == 20
    assert not result["same_exposure"]
    assert not result["baseline"]["half_second_observed"]
    assert not result["candidate"]["half_second_observed"]
    assert not result["clean_reception_certified"]


def test_requires_same_physics_and_executed_action_memory():
    b, c = sample(30), sample(30)
    c["qpos"][0, 0, 0] = 1
    with pytest.raises(ValueError):
        compare_long_pair(b, c, layout=LAYOUT)
    c = sample(30)
    c["executed_receiver_action"][0, 0] = 1
    with pytest.raises(ValueError):
        compare_long_pair(b, c, layout=LAYOUT)
