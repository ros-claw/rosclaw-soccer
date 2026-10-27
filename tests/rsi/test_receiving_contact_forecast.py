import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_contact_forecast import future_first_contact_labels


def test_first_event_in_window_and_right_censoring():
    force = np.zeros((9, 10, 2, 3), dtype=np.float64)
    force[3, 7, 1, 0] = 2
    force[4, 0, 1, 2] = 2
    force[7, 0, 1, (0, 2)] = 2
    assert future_first_contact_labels(force, horizon=3).tolist() == [0, 1, 1, 1, 2, 2, 2]
    assert len(future_first_contact_labels(force, horizon=3)) == 7


def test_exact_threshold_and_invalid_horizon():
    force = np.zeros((5, 10, 2, 3), dtype=np.float64)
    force[0, 0, 1, 0] = 1
    assert future_first_contact_labels(force).tolist() == [0]
    with pytest.raises(ValueError):
        future_first_contact_labels(force, horizon=6)
