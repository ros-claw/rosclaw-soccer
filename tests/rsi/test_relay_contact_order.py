import numpy as np
import pytest

from rosclaw_soccer.rsi.relay_contact_order import relay_contact_order


def test_strict_substep_order_and_handoff():
    f = np.zeros((3, 10, 2, 3))
    f[0, 1, 0, 0] = 11
    f[1, 1, 1, 2] = 2
    assert relay_contact_order(f, 2)["ordered_source_then_receiver_before_handoff"]
    assert not relay_contact_order(f, 1)["ordered_source_then_receiver_before_handoff"]
    assert not relay_contact_order(f, None)["causal_attribution_proven"]


def test_nonfinite_empty_and_bad_handoff_rejected():
    f = np.zeros((1, 10, 2, 3))
    with pytest.raises(ValueError):
        relay_contact_order(f, True)
    f[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        relay_contact_order(f, 0)
