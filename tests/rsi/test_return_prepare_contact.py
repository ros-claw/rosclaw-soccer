import numpy as np
import pytest

from rosclaw_soccer.rsi.return_prepare_contact import return_prepare_contact


def test_post_handoff_order_and_common_horizon():
    f = np.zeros((5, 10, 2, 3), np.float64)
    failed = np.zeros((5, 2), bool)
    f[0, 0, 1, 0] = 20  # Pre-handoff contact must not trigger return.
    f[2, 0, 1, 0] = 20
    f[3, 2, 0, 0] = 20
    f[4, 0, 0, 2] = 2
    full = return_prepare_contact(f, failed, handoff=1, outgoing_foot_strike=True)
    assert full["post_handoff_b_foot_substep"] == 20
    assert full["a_foot_before_nonfoot"] and not full["a_clean_foot_contact"]
    assert not full["possession_proven"]
    short = return_prepare_contact(f[:4], failed[:4], handoff=1, outgoing_foot_strike=True)
    assert short["a_clean_foot_contact"]
    failed[3, 1] = True
    assert not return_prepare_contact(f[:4], failed[:4], handoff=1, outgoing_foot_strike=True)[
        "a_clean_foot_contact"
    ]


def test_rejects_bad_handoff_and_nonfinite_forces():
    f = np.zeros((2, 10, 2, 3))
    failed = np.zeros((2, 2), bool)
    with pytest.raises(ValueError):
        return_prepare_contact(f, failed, handoff=2, outgoing_foot_strike=False)
    f[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        return_prepare_contact(f, failed, handoff=0, outgoing_foot_strike=False)
