import numpy as np
import pytest

from rosclaw_soccer.rsi.dual_receive_geometry_audit import first_receiver_nonfoot_after_pass


def test_receiver_contact_must_be_strictly_after_passer_foot():
    forces = np.zeros((4, 2, 3))
    forces[1, 0, 0] = 11
    forces[1, 1, 2] = 2
    forces[3, 1, 2] = 3
    assert first_receiver_nonfoot_after_pass(forces) == (1, 3)
    forces[3, 1, 2] = 0
    assert first_receiver_nonfoot_after_pass(forces) == (1, None)


def test_geometry_audit_rejects_nonfinite_and_negative_force():
    forces = np.zeros((4, 2, 3))
    forces[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        first_receiver_nonfoot_after_pass(forces)
    forces[0, 0, 0] = -1
    with pytest.raises(ValueError):
        first_receiver_nonfoot_after_pass(forces)
