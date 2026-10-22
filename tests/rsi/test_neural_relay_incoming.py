import numpy as np
import pytest

from rosclaw_soccer.rsi.neural_relay_incoming import incoming_delivery


def test_receiver_cannot_retroactively_improve_incoming_delivery_with_later_dribble():
    ball = np.zeros((10, 3))
    ball[:, 0] = np.arange(10)
    forces = np.zeros((10, 2, 3))
    forces[1, 0, 0] = 20
    forces[5, 1, 2] = 3
    result = incoming_delivery(ball, forces, np.array([9.0, 0, 0]))
    assert result["A_first_foot_strike_substep"] == 1
    assert result["B_first_ball_contact_substep"] == 5
    assert result["incoming_target_error_3d_before_B_contact_m"] == 5
    assert not result["incoming_diagnostic_is_received_pass_qualification"]


def test_no_source_strike_means_unavailable_not_fabricated_zero_error():
    result = incoming_delivery(np.zeros((10, 3)), np.zeros((10, 2, 3)), np.zeros(3))
    assert not result["incoming_error_observed"]
    assert result["incoming_target_error_3d_before_B_contact_m"] is None


def test_no_receiver_contact_preserves_observed_delivery_without_calling_reception_success():
    ball = np.zeros((10, 3))
    ball[:, 0] = np.arange(10)
    forces = np.zeros((10, 2, 3))
    forces[1, 0, 0] = 20
    result = incoming_delivery(ball, forces, np.array([9.0, 0, 0]))
    assert result["incoming_target_error_3d_before_B_contact_m"] == 0
    assert result["B_first_ball_contact_substep"] is None


def test_nonfinite_or_negative_force_is_invalid():
    f = np.zeros((10, 2, 3))
    f[0, 0, 0] = -1
    with pytest.raises(ValueError):
        incoming_delivery(np.zeros((10, 3)), f, np.zeros(3))
