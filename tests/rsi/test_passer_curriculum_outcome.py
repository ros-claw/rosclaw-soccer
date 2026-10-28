import numpy as np

from rosclaw_soccer.rsi.passer_curriculum_outcome import strict_pass_receive


def test_clean_receiver_requires_order_no_body_contact_and_health():
    forces = np.zeros((2, 10, 2, 3))
    failed = np.zeros((2, 2), dtype=np.bool_)
    forces[0, 3, 0, 0] = 11
    forces[1, 2, 1, 1] = 11
    assert (
        strict_pass_receive(forces, failed, handoff=True)["strict_receiver_clean_handoff"] is True
    )
    forces[1, 3, 1, 2] = 1.1
    assert (
        strict_pass_receive(forces, failed, handoff=True)["strict_receiver_clean_handoff"] is False
    )
    forces[1, 3, 1, 2] = 0
    failed[1, 0] = True
    assert (
        strict_pass_receive(forces, failed, handoff=True)["strict_receiver_clean_handoff"] is False
    )
    failed[1, 0] = False
    forces[0, 1, 1, 0] = 11
    assert (
        strict_pass_receive(forces, failed, handoff=True)["strict_receiver_clean_handoff"] is False
    )
