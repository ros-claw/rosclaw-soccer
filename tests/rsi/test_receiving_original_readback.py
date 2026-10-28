import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_original_readback import original_reception_readback


def test_readback_requires_actual_contact_and_15_consecutive_controls():
    q = np.zeros((16, 10, 79))
    v = np.zeros((16, 10, 76))
    f = np.zeros((16, 10, 2, 3))
    failed = np.zeros((16, 2), dtype=np.bool_)
    q[:, :, 2], q[:, :, 3], q[:, :, 72], q[:, :, 74] = 0.8, 1, 0.3, 0.11
    args = dict(receiver_q=0, receiver_v=0, ball_q=72, ball_v=70, excluded_contact_ever=False)
    assert original_reception_readback(q, v, f, failed, **args)["first_ready_control"] is None
    f[0, 0, 1, 0] = 2
    assert original_reception_readback(q, v, f, failed, **args)["first_ready_control"] == 15
    assert (
        original_reception_readback(q[:14], v[:14], f[:14], failed[:14], **args)[
            "first_ready_control"
        ]
        is None
    )
    q[7, :, 72] = 2
    assert original_reception_readback(q, v, f, failed, **args)["first_ready_control"] is None
    assert (
        original_reception_readback(q, v, f, failed, **(args | {"excluded_contact_ever": True}))[
            "reconstruction_exact"
        ]
        is False
    )
    with pytest.raises(ValueError):
        original_reception_readback(q, v, f, failed, **(args | {"ball_q": 0}))
