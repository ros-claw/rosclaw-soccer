import numpy as np
import pytest

from rosclaw_soccer.rsi.passer_curriculum_outcome import delivery_center_speeds, strict_pass_receive


def test_delivery_speeds_use_post_a_and_pre_b_samples():
    velocities = np.array([[3.0, 0, 0], [0.6, 0.8, 0], [9.0, 0, 0]])
    result = delivery_center_speeds(velocities, passer_substep=0, receiver_substep=2)
    assert result["ball_center_speed_after_passer_mps"] == 3
    assert result["ball_center_speed_before_receiver_mps"] == 1
    assert result["speed_is_not_contact_point_relative_velocity"] is True
    assert (
        delivery_center_speeds(velocities, passer_substep=2, receiver_substep=2)[
            "ball_center_speed_before_receiver_mps"
        ]
        is None
    )
    with pytest.raises(ValueError):
        delivery_center_speeds(velocities, passer_substep=-1, receiver_substep=2)
    with pytest.raises(ValueError):
        delivery_center_speeds(
            np.array([[float("nan"), 0, 0]]), passer_substep=0, receiver_substep=None
        )


def test_clean_receiver_requires_order_no_body_contact_and_health():
    forces = np.zeros((2, 10, 2, 3))
    failed = np.zeros((2, 2), dtype=np.bool_)
    forces[0, 3, 0, 0] = 11
    forces[1, 2, 1, 1] = 11
    assert (
        strict_pass_receive(forces, failed, handoff=True)["strict_receiver_clean_handoff"] is True
    )
    assert strict_pass_receive(forces, failed, handoff=True)["strict_both_clean_handoff"] is True
    forces[0, 4, 0, 2] = 1.1
    result = strict_pass_receive(forces, failed, handoff=True)
    assert result["strict_receiver_clean_handoff"] is True
    assert result["strict_both_clean_handoff"] is False
    assert result["passer_nonfoot_substep"] == 4
    forces[0, 4, 0, 2] = 0
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
