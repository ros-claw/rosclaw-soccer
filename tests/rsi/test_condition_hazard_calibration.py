import numpy as np
import pytest

from rosclaw_soccer.rsi.condition_hazard_calibration import (
    calibrate_negative_maxima,
    condition_hazard_summary,
)


def test_condition_maxima_and_calibration_do_not_count_frames_as_trials():
    rows = condition_hazard_summary(
        np.array([0.1, 0.8, 0.4, 0.9]), np.array([0, 0, 2, 0]), np.array([1, 1, 2, 3])
    )
    assert rows == [
        dict(condition=1, max_risk=0.8, contains_nonfoot=False),
        dict(condition=2, max_risk=0.4, contains_nonfoot=True),
        dict(condition=3, max_risk=0.9, contains_nonfoot=False),
    ]
    threshold, count = calibrate_negative_maxima(rows, alpha=0.5)
    assert count == 2 and threshold == 0.9


def test_insufficient_calibration_fails_closed_without_fabricated_precision():
    rows = [dict(condition=1, max_risk=0.2, contains_nonfoot=False)]
    assert calibrate_negative_maxima(rows, alpha=0.1) == (1.0, 1)
    assert calibrate_negative_maxima([], alpha=0.1) == (1.0, 0)


def test_duplicate_or_nonfinite_conditions_rejected():
    with pytest.raises(ValueError):
        calibrate_negative_maxima(
            [dict(condition=1, max_risk=0.2, contains_nonfoot=False)] * 2, alpha=0.1
        )
    with pytest.raises(ValueError):
        condition_hazard_summary(np.array([np.nan]), np.array([0]), np.array([1]))
