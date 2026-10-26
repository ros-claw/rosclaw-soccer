import numpy as np
import pytest

from rosclaw_soccer.data.motion_anchor_guard import MotionAnchorGuard


def test_detects_one_frame_shift_and_does_not_mutate_training_clip():
    first = np.arange(360, dtype=np.float64).reshape(120, 3) / 100
    before = first.copy()
    held = np.vstack([[-1.0, -1.0, -1.0], first.copy()])
    held[30, 0] += 1e-7
    guard = MotionAnchorGuard({"train": first})
    with pytest.raises(ValueError, match="overlap across splits"):
        guard.inspect("retention", held)
    np.testing.assert_array_equal(first, before)


def test_no_hits_is_not_a_proof_of_independence():
    train = np.arange(360, dtype=np.float64).reshape(120, 3)
    guard = MotionAnchorGuard({"train": train})
    result = guard.inspect("retention", train * 2 + 1)
    assert result["detected_shared_segments"] == 0
    assert result["source_independence_proven"] is False
    assert result["possible_unanchored_near_duplicates_not_excluded"] is True


def test_excess_repeated_anchors_fail_closed():
    train = np.zeros((120, 3), dtype=np.float64)
    guard = MotionAnchorGuard({"train": train}, stride=1, max_anchor_hits=10)
    with pytest.raises(ValueError, match="hit budget exhausted"):
        guard.inspect("retention", train.copy())


def test_bad_input_and_index_budget_fail_closed():
    with pytest.raises(ValueError, match="index budget exhausted"):
        MotionAnchorGuard({"train": np.zeros((120, 3))}, stride=1, max_index_entries=1)
    guard = MotionAnchorGuard({"train": np.zeros((120, 3))})
    bad = np.zeros((120, 3))
    bad[1, 0] = np.nan
    with pytest.raises(ValueError):
        guard.inspect("retention", bad)
