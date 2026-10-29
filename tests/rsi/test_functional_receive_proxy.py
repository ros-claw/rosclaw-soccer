import numpy as np
import pytest

from rosclaw_soccer.rsi.functional_receive_proxy import functional_receive_proxy


def test_contact_window_is_causal_and_failure_still_penalized():
    r, b, f = np.zeros((20, 2)), np.zeros((20, 2)), np.zeros((20, 10, 3))
    f[2, 1, 2] = 2
    reward, window = functional_receive_proxy(r, b, f, np.zeros(20, bool), eventual_failure=True)
    assert window == 11
    assert reward[11] == 1 and reward[-1] == -2 and reward.sum() == -1


def test_no_contact_or_short_window_never_receives_bonus():
    r, b, f = np.zeros((9, 2)), np.zeros((9, 2)), np.zeros((9, 10, 3))
    reward, window = functional_receive_proxy(r, b, f, np.zeros(9, bool), eventual_failure=False)
    assert window is None and not reward.any()
    f[0, 0, 0] = 2
    reward, window = functional_receive_proxy(r, b, f, np.zeros(9, bool), eventual_failure=False)
    assert window is None and not reward.any()


def test_nan_rejected_instead_of_rewarding():
    r = np.zeros((2, 2))
    r[0, 0] = np.nan
    with pytest.raises(ValueError):
        functional_receive_proxy(
            r, np.zeros((2, 2)), np.zeros((2, 10, 3)), np.zeros(2, bool), eventual_failure=False
        )
