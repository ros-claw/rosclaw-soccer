import numpy as np
import pytest

from scripts.rsi_analyze_memory_contact_failures import contact_times


def test_native_timing_does_not_round_early_nonfoot_into_later_foot():
    force = np.zeros((20, 1, 3))
    force[1, 0, 2] = 1.01
    force[9, 0, 0] = 1.0  # Frozen threshold is strictly greater than 1 N.
    force[10, 0, 0] = 2.0
    force[19, 0, 1] = 3.0
    assert contact_times(force) == dict(left=10, right=19, nonfoot=1)


def test_absent_contact_is_not_synthetic_zero_time():
    assert contact_times(np.zeros((3600, 1, 3))) == dict(left=None, right=None, nonfoot=None)


@pytest.mark.parametrize(
    "forces",
    [
        np.zeros((0, 1, 3)),
        np.zeros((3601, 1, 3)),
        np.zeros((10, 3)),
        np.zeros((10, 2, 3)),
        np.full((10, 1, 3), -1.0),
        np.full((10, 1, 3), np.nan),
        np.full((10, 1, 3), np.inf),
    ],
)
def test_invalid_force_trace_rejected(forces):
    with pytest.raises(ValueError):
        contact_times(forces)
