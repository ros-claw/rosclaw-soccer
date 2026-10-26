import numpy as np
import pytest

from rosclaw_soccer.data.motion_overlap import motion_overlap, require_no_cross_split_overlap


def test_detects_shifted_near_duplicate_without_mutation():
    first = np.arange(240, dtype=np.float64).reshape(120, 2)
    second = np.vstack([[-100.0, -100.0], first.copy()])
    second[30, 0] += 1e-7
    before = second.copy()
    result = motion_overlap(first, second)
    assert result["near_duplicate"] is True
    assert result["shift_frames"] == 1
    assert result["overlap_frames"] == 120
    assert result["source_independence_proven"] is False
    np.testing.assert_array_equal(before, second)


def test_independent_values_do_not_certify_independence():
    first = np.arange(240, dtype=np.float64).reshape(120, 2)
    second = first * 2
    result = motion_overlap(first, second)
    assert result["near_duplicate"] is False
    assert result["source_independence_proven"] is False


def test_rejects_nonfinite_or_incompatible_inputs():
    x = np.zeros((120, 2), dtype=np.float64)
    y = x.copy()
    y[0, 0] = np.nan
    with pytest.raises(ValueError):
        motion_overlap(x, y)
    with pytest.raises(ValueError):
        motion_overlap(x, x[:, :1])


def test_cross_split_guard_rejects_shifted_duplicate():
    first = np.arange(240, dtype=np.float64).reshape(120, 2)
    second = np.vstack([[-100.0, -100.0], first.copy()])
    with pytest.raises(ValueError, match="overlap across splits"):
        require_no_cross_split_overlap({"a": first, "b": second}, {"a": "train", "b": "fresh"})


def test_cross_split_guard_does_not_assert_independence():
    first = np.arange(240, dtype=np.float64).reshape(120, 2)
    result = require_no_cross_split_overlap(
        {"a": first, "b": first * 2}, {"a": "train", "b": "fresh"}
    )
    assert result["cross_split_pairs_examined"] == 1
    assert result["source_independence_proven"] is False
