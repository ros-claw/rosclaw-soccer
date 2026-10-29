import numpy as np
import pytest

from rosclaw_soccer.rsi.team_return_interaction import return_interaction


def test_return_window_requires_strike_and_actual_contact():
    q, v, f, failure = (
        np.zeros((120, 9)),
        np.zeros((120, 9)),
        np.zeros((120, 2, 3)),
        np.zeros((12, 2), bool),
    )
    v[:, 0] = 1.1
    f[1, 1, 0] = 11
    args = dict(start=0, direction=np.array([1.0, 0.0]), ball_q=0, ball_v=0, player_q=3, player_v=3)
    assert return_interaction(q, v, f, failure, **args)["near_healthy_longest_substeps"] == 0
    f[3, 0, 2] = 2
    v[3:, 0] = 0
    out = return_interaction(q, v, f, failure, **args)
    assert out["first_A_nonfoot_step"] == 3 and out["near_healthy_longest_substeps"] == 117
    failure[5, 0] = True
    assert return_interaction(q, v, f, failure, **args)["near_healthy_longest_substeps"] == 60
    with pytest.raises(ValueError):
        return_interaction(q, v, f, failure.astype(float), **args)
