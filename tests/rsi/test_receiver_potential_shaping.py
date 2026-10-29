import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_potential_shaping import receiver_potential_shaping


def test_potential_is_causal_bounded_and_telescopes_even_at_segment_end():
    q = np.zeros((4, 10))
    v = np.zeros((4, 10))
    q[:, 5] = [2, 1, 0.5, 0.4]
    v[:, 5] = [1, 0.8, 0.5, 0.2]
    opts = dict(receiver_q=0, receiver_v=0, ball_q=5, ball_v=5)
    reward = receiver_potential_shaping(q, v, **opts)
    expected_start = 0.5 * np.exp(-2 / 0.65 - 1 / 1.5)
    assert abs(sum(0.99**i * x for i, x in enumerate(reward)) + expected_start) < 1e-12
    assert np.max(np.abs(reward)) <= 0.5
    changed = q.copy()
    changed[-1, 5] = 10
    revised = receiver_potential_shaping(changed, v, **opts)
    np.testing.assert_array_equal(reward[:2], revised[:2])
    assert reward[2] != revised[2]


def test_rejects_invalid_or_nonfinite_physics():
    opts = dict(receiver_q=0, receiver_v=0, ball_q=5, ball_v=5)
    with pytest.raises(ValueError):
        receiver_potential_shaping(np.zeros((0, 10)), np.zeros((0, 10)), **opts)
    q = np.zeros((2, 10))
    q[0, 5] = np.nan
    with pytest.raises(ValueError):
        receiver_potential_shaping(q, np.zeros((2, 10)), **opts)
