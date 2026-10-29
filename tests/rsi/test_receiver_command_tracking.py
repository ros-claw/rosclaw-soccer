import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_command_tracking import receiver_command_tracking


def test_body_frame_rotation_and_exclude_post_contact():
    q = np.tile(np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)]), (2, 1))
    command = np.array([[1.0, 0, 0], [0.0, 0, 0]])
    velocity = np.array([[0.0, 1.0, 0], [5.0, 0, 0]])
    flags = np.array([[False, False], [True, False]])
    result = receiver_command_tracking(command, q, velocity, flags)
    assert result["precontact_observations"] == 1
    assert result["precontact_velocity_error_mean_mps"] == pytest.approx(0, abs=1e-12)
    assert result["reachability_proven"] is False


def test_contact_latch_must_not_forget():
    with pytest.raises(ValueError):
        receiver_command_tracking(
            np.zeros((2, 3)),
            np.tile([1.0, 0, 0, 0], (2, 1)),
            np.zeros((2, 3)),
            np.array([[True, False], [False, False]]),
        )
