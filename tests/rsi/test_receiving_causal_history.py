import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_causal_history import augment_foot_with_velocity_history


def test_history_is_past_only_and_padding_is_explicit():
    foot = np.zeros((5, 12), dtype=np.float64)
    obs = np.broadcast_to(np.arange(5, dtype=np.float32)[:, None], (5, 98)).copy()
    out = augment_foot_with_velocity_history(foot, obs)
    assert out.shape == (5, 308)
    assert out[:, 306:].tolist() == [[0, 0], [1, 0], [1, 1], [1, 1], [1, 1]]
    assert (out[2, 12:110] == 2).all()
    assert (out[2, 110:208] == 1).all()
    assert (out[2, 208:306] == 0).all()
    obs[3:] = 999
    assert np.array_equal(out[:3], augment_foot_with_velocity_history(foot, obs)[:3])
    with pytest.raises(ValueError):
        augment_foot_with_velocity_history(foot[:-1], obs)
