import numpy as np

from rosclaw_soccer.rsi.relay_teammate_successor import teammate_successor


def test_relative_state_and_right_censoring_after_ordered_foot_contact():
    q, v, f = np.zeros((4, 10, 8)), np.zeros((4, 10, 8)), np.zeros((4, 10, 2, 3))
    f[1, 0, 1, 0] = 2
    f[1, 2, 0, 0] = 2
    q[1, 2, 0] = 0.4
    v[1, 2, 0] = 0.2
    result = teammate_successor(
        q, v, f, return_start=1, ball_q=0, ball_v=0, teammate_q=4, teammate_v=4, timestep=0.002
    )
    assert result["status"] == "ORDERED_FOOT_CONTACT_NOT_POSSESSION"
    assert result["samples"][0]["ball_to_pelvis_planar_m"] == 0.4
    assert result["samples"][0]["relative_planar_speed_mps"] == 0.2
    assert result["samples"][1]["status"] == "RIGHT_CENSORED"
