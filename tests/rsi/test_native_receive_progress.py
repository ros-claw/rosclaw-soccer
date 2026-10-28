import numpy as np
import pytest

from rosclaw_soccer.rsi.native_receive_credit import native_control_progress


def test_native_progress_counts_only_controlled_substeps_and_eligible_worlds():
    distance = np.full((3, 10), 0.3)
    speed = np.full((3, 10), 0.4)
    speed[0, 5:] = 1.0
    distance[1, 2:] = 0.7
    progress = native_control_progress(distance, speed, np.array([True, True, False]))
    np.testing.assert_allclose(progress, [0.5, 0.2, 0.0])


def test_native_progress_rejects_bad_measurements():
    distance = np.zeros((1, 10))
    speed = np.zeros((1, 10))
    speed[0, 3] = np.nan
    with pytest.raises(ValueError):
        native_control_progress(distance, speed, np.array([True]))
    with pytest.raises(ValueError):
        native_control_progress(distance, np.zeros((1, 10)), np.array([1]))
