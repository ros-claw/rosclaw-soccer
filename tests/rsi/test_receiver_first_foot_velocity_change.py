import numpy as np
import pytest

from rosclaw_soccer.rsi.passer_curriculum_outcome import receiver_first_foot_velocity_change


def test_one_substep_change_is_measured_and_not_called_isolated_impulse():
    forces = np.zeros((2, 10, 2, 3), np.float64)
    velocity = np.zeros((2, 10, 3), np.float64)
    forces[0, 5, 1, 0] = 20
    forces[0, 5, 1, 2] = 2
    velocity[0, 4] = [3, 0, 0]
    velocity[0, 5] = [2, 0, 0]
    result = receiver_first_foot_velocity_change(forces, velocity)
    assert result["first_foot_substep"] == 5
    assert result["pre_speed_mps"] == 3
    assert result["post_speed_mps"] == 2
    assert result["velocity_change_mps"] == 1
    assert result["concurrent_nonfoot"]
    assert not result["isolated_foot_impulse"]


def test_missing_contact_and_first_sample_without_predecessor():
    forces = np.zeros((1, 10, 2, 3), np.float64)
    velocity = np.zeros((1, 10, 3), np.float64)
    assert receiver_first_foot_velocity_change(forces, velocity)["first_foot_substep"] is None
    forces[0, 0, 1, 1] = 11
    result = receiver_first_foot_velocity_change(forces, velocity)
    assert result["first_foot_substep"] == 0
    assert not result["pre_sample_available"]
    assert result["pre_speed_mps"] is None and result["velocity_change_mps"] is None


def test_invalid_or_misaligned_samples_fail_closed():
    forces = np.zeros((1, 10, 2, 3), np.float64)
    velocity = np.zeros((1, 10, 3), np.float64)
    velocity[0, 2, 0] = np.nan
    with pytest.raises(ValueError):
        receiver_first_foot_velocity_change(forces, velocity)
    with pytest.raises(ValueError):
        receiver_first_foot_velocity_change(forces, np.zeros((10, 3)))
