import numpy as np
import pytest

from rosclaw_soccer.rsi.reference_rate_projection import project_rate


def test_projection_enforces_bound_without_mutating_input_or_endpoints():
    offsets = np.zeros((4, 6), dtype=np.float64)
    offsets[1:3, 0] = 0.2
    before = offsets.copy()
    result, scale = project_rate(offsets, max_step=0.08)
    np.testing.assert_array_equal(offsets, before)
    np.testing.assert_array_equal(result[[0, -1]], 0)
    assert scale == pytest.approx(0.4)
    assert np.max(np.abs(np.diff(result, axis=0))) <= 0.08


def test_projection_leaves_feasible_offsets_unchanged():
    offsets = np.zeros((4, 6), dtype=np.float64)
    offsets[1:3, 0] = 0.05
    result, scale = project_rate(offsets, max_step=0.08)
    assert scale == 1.0
    np.testing.assert_array_equal(result, offsets)


@pytest.mark.parametrize("bound", [0.0, -1.0, float("nan"), 1, True])
def test_projection_rejects_invalid_bound(bound):
    with pytest.raises(ValueError):
        project_rate(np.zeros((4, 6)), max_step=bound)
