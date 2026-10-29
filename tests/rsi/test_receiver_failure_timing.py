import numpy as np
import pytest

from scripts.rsi_export_receiver_failure_timing import failure_timing


def test_failure_timing_uses_first_actual_control_and_all_actors():
    failed = np.zeros((9, 2), dtype=bool)
    failed[5, 1] = True
    failed[7, 0] = True
    result = failure_timing(failed, 4, dict(body_failure=True, A_failed=True, B_failed=True))
    assert result == dict(failure_control=5, failed_bodies=["A", "B"], physical_controls=9)


def test_rejects_mismatch_and_truncation():
    failed = np.zeros((3, 2), dtype=bool)
    with pytest.raises(ValueError):
        failure_timing(failed, 4, dict(body_failure=False, A_failed=False, B_failed=False))
    with pytest.raises(ValueError):
        failure_timing(failed, 3, dict(body_failure=True, A_failed=False, B_failed=False))
