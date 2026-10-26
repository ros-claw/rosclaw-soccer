import numpy as np
import pytest

from rosclaw_soccer.rsi.cftrack_history_observation import (
    HISTORY_SLICES,
    assemble_proprioception,
    flatten_five_frame_history,
)


def test_proprioception_source_order_and_permutation():
    parts = [
        np.full(n, float(i), dtype=np.float64)
        for i, n in enumerate((15, 3, 3, 29, 29, 29, 3, 6, 24))
    ]
    parts[3] = np.arange(29, dtype=np.float64)
    parts[4] = np.arange(29, dtype=np.float64) + 30
    perm = np.arange(29)[::-1]
    result = assemble_proprioception(*parts, mj2lab=perm, default_lab=np.ones(29, np.float32))
    assert result.dtype == np.float32 and result.shape == (141,)
    np.testing.assert_array_equal(result[21:50], perm - 1)
    np.testing.assert_array_equal(result[50:79], perm + 30)
    np.testing.assert_array_equal(result[79:108], np.full(29, 5))
    np.testing.assert_array_equal(result[117:141], np.full(24, 8))


def test_history_source_feature_major_not_frame_major():
    prior = np.arange(4 * 141, dtype=np.float32).reshape(4, 141)
    current = np.arange(4 * 141, 5 * 141, dtype=np.float32)
    previous = prior.copy()
    result = flatten_five_frame_history(prior, current)
    h = np.vstack((prior, current))
    expected = np.concatenate([h[:, start:stop].ravel() for start, stop in HISTORY_SLICES])
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(prior, previous)
    assert result.shape == (705,)


def test_history_rejects_nonfinite_or_missing_samples():
    prior = np.zeros((4, 141), np.float32)
    current = np.zeros(141, np.float32)
    prior[0, 0] = np.nan
    with pytest.raises(ValueError):
        flatten_five_frame_history(prior, current)
    with pytest.raises(ValueError):
        flatten_five_frame_history(np.zeros((3, 141), np.float32), current)
