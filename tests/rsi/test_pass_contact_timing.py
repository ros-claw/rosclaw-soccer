import numpy as np
import pytest

from rosclaw_soccer.rsi.pass_contact_timing import contact_timing


def test_strict_thresholds_and_substep_delay():
    f = np.zeros((3, 10, 2, 3))
    f[0, 5, 1, 0] = 10
    f[1, 2, 1, 1] = 10.01
    f[1, 5, 1, 2] = 1.01
    result = contact_timing(f, 0.002)
    assert result["first_foot_substep"] == 12
    assert result["first_nonfoot_substep"] == 15
    assert result["foot_first"]
    assert result["delay_s"] == pytest.approx(0.006)


def test_absent_simultaneous_and_malformed():
    f = np.zeros((1, 10, 2, 3))
    assert contact_timing(f, 0.002)["delay_s"] is None
    f[0, 1, 1] = [11, 0, 2]
    assert contact_timing(f, 0.002)["foot_first"] is False
    f[0, 1, 1, 2] = np.nan
    with pytest.raises(ValueError):
        contact_timing(f, 0.002)
