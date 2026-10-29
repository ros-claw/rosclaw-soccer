import numpy as np
import pytest

from rosclaw_soccer.rsi.relay_return_evidence import return_observation


def test_target_hit_and_teammate_contact_are_distinct_observations():
    q = np.zeros((4, 6))
    q[:, 0] = [3, 2, 0, 1]
    q[:, 3] = 0.5
    v = np.zeros((4, 3))
    f = np.zeros((4, 2, 3))
    f[3, 0, 2] = 3
    r = return_observation(q, v, f, start=1, goal=[0, 0, 0], ball_q=0, ball_v=0, teammate_q=3)
    assert r["closest_step"] == 2
    assert r["closest_error_m"] == 0
    assert r["teammate_planar_distance_m"] == 0.5
    assert r["teammate_first_nonfoot_over_1n_step"] == 3
    assert r["teammate_first_foot_over_10n_step"] is None
    assert r["teammate_possession_proven"] is False


def test_rejects_unaligned_or_nonfinite_trajectories():
    q = np.zeros((2, 6))
    q[0, 0] = np.nan
    with pytest.raises(ValueError):
        return_observation(
            q,
            np.zeros((2, 3)),
            np.zeros((2, 2, 3)),
            start=0,
            goal=[0, 0, 0],
            ball_q=0,
            ball_v=0,
            teammate_q=3,
        )
