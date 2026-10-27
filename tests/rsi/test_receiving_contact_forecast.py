import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_contact_forecast import future_first_contact_labels
from scripts.rsi_audit_continuous_contact_forecast import independent_first_label


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


def test_independent_window_label_matches_source_at_every_control():
    rng = np.random.default_rng(19)
    force = np.zeros((24, 10, 2, 3), dtype=np.float64)
    force[:, :, 1, :] = (rng.random((24, 10, 3)) > 0.98) * 2
    labels = future_first_contact_labels(force)
    assert labels.tolist() == [
        independent_first_label(force[t : t + 5, :, 1, :]) for t in range(len(labels))
    ]
