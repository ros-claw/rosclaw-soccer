import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_reference_blend import blend_planner_segment


def segments():
    old = np.zeros((30, 36), dtype=np.float64)
    old[:, 3] = 1
    old[:, 7] = np.arange(30) * 0.01
    new = old.copy()
    new[:, :3] += 0.2
    new[:, 7:] += 0.12
    new[:, 3:7] = [-1, 0, 0, 0]
    return old, new


def test_fixed_endpoints_tail_ownership_and_shortest_rotation():
    old, new = segments()
    saved = old.copy(), new.copy()
    blended = blend_planner_segment(old, new)
    np.testing.assert_array_equal(blended[0], old[0])
    np.testing.assert_array_equal(blended[9:], new[9:])
    np.testing.assert_allclose(abs(blended[:, 3]), 1, atol=0, rtol=0)
    np.testing.assert_array_equal(blended[:, 4:7], np.zeros((30, 3)))
    t = np.arange(10) / 9
    w = t**3 * (10 - 15 * t + 6 * t * t)
    np.testing.assert_allclose(blended[:10, 7], old[:10, 7] + w * 0.12, atol=1e-16)
    assert not blended.flags.writeable
    for a, b in zip(saved, (old, new), strict=True):
        np.testing.assert_array_equal(a, b)
        assert not np.shares_memory(blended, b)


def test_identical_segments_preserve_every_bit():
    old, _ = segments()
    np.testing.assert_array_equal(blend_planner_segment(old, old.copy()), old)


@pytest.mark.parametrize("kind", ["dtype", "nan", "shape", "length", "quaternion"])
def test_invalid_segments_not_silently_repaired(kind):
    old, new = segments()
    if kind == "dtype":
        new = new.astype(np.float32)
    elif kind == "nan":
        new[20, 2] = np.nan
    elif kind == "shape":
        new = new[:, :35]
    elif kind == "length":
        old, new = old[:9], new[:9]
    else:
        new[15, 3] = 0.1
    with pytest.raises(ValueError):
        blend_planner_segment(old, new)
