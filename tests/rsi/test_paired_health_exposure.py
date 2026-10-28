import numpy as np
import pytest

from rosclaw_soccer.rsi.paired_health_exposure import paired_health_at_common_horizon


def test_failure_after_shorter_trajectory_is_not_a_common_horizon_regression():
    a = np.zeros((300, 2), dtype=bool)
    b = np.zeros((520, 2), dtype=bool)
    b[519, 0] = True
    row = paired_health_at_common_horizon(a, b)
    assert row["common_controls"] == 300 and not row["equal_exposure"]
    assert row["raw_healthy_common"] and row["executed_healthy_common"]
    assert row["raw_healthy_full"] and not row["executed_healthy_full"]


def test_common_horizon_preserves_actual_body_and_failure():
    a = np.zeros((4, 2), dtype=bool)
    b = np.zeros((6, 2), dtype=bool)
    b[2, 1] = True
    row = paired_health_at_common_horizon(a, b)
    assert not row["executed_healthy_common"]
    assert row["executed_failure_by_body_common"] == [False, True]


@pytest.mark.parametrize("bad", [np.zeros((2, 2)), np.zeros((0, 2), bool), np.zeros((2, 3), bool)])
def test_invalid_history_fails_closed(bad):
    with pytest.raises(ValueError):
        paired_health_at_common_horizon(bad, np.zeros((2, 2), bool))
