import numpy as np
import pytest

from rosclaw_soccer.rsi.relay_common_successor import common_successor


def test_common_exposure_censors_contact_beyond_shorter_trial():
    h = dict(
        qpos=np.zeros((40, 10, 79)),
        qvel=np.zeros((40, 10, 76)),
        ball_forces=np.zeros((40, 10, 2, 3)),
    )
    h["ball_forces"][10, 0, 1, 0] = 2
    h["ball_forces"][30, 0, 0, 0] = 2
    layout = dict(ball_q=0, ball_v=0, teammate_q=7, teammate_v=6)
    short = common_successor(h, common=20, return_start=5, layout=layout, timestep=0.002)
    full = common_successor(h, common=40, return_start=5, layout=layout, timestep=0.002)
    assert short["status"] == "NO_LATER_TEAMMATE_FOOT_CONTACT"
    assert full["status"] == "ORDERED_FOOT_CONTACT_NOT_POSSESSION"
    assert full["samples"][-1]["status"] == "RIGHT_CENSORED"
    assert (
        common_successor(h, common=20, return_start=20, layout=layout, timestep=0.002)["status"]
        == "RETURN_START_RIGHT_CENSORED"
    )
    with pytest.raises(ValueError):
        common_successor(h, common=41, return_start=0, layout=layout, timestep=0.002)
