import numpy as np
import pytest

from rosclaw_soccer.rsi.relay_follow_prefix_audit import verify_follow_prefix


def case():
    return {
        "qpos": np.zeros((6, 10, 9)),
        "qvel": np.zeros((6, 10, 8)),
        "torque": np.zeros((6, 10, 6)),
        "ball_forces": np.zeros((6, 10, 2, 3)),
    }


def test_next_control_after_real_return_foot_and_exact_prefix():
    a, b = case(), case()
    a["ball_forces"][3, 1, 1, 0] = b["ball_forces"][3, 1, 1, 0] = 2
    b.update(
        passer_follow_control=np.array([144, 145]),
        passer_follow_action=np.zeros((2, 29)),
        passer_follow_command=np.zeros((2, 3)),
        passer_follow_observation=np.zeros((2, 98)),
        passer_follow_target=np.zeros((2, 29)),
    )
    b["qpos"][4, 0, 0] = 1
    result = verify_follow_prefix(a, b, initial_control=140, receiver_velocity_controls=2)
    assert result["exact_prefix_controls"] == 4
    assert result["follow_controls"] == 2
    b["qpos"][3, 0, 0] = 1
    with pytest.raises(ValueError, match="pre-follow physics"):
        verify_follow_prefix(a, b, initial_control=140, receiver_velocity_controls=2)


def test_reject_follow_without_actual_trigger():
    a, b = case(), case()
    b.update(
        passer_follow_control=np.array([144]),
        passer_follow_action=np.zeros((1, 29)),
        passer_follow_command=np.zeros((1, 3)),
        passer_follow_observation=np.zeros((1, 98)),
        passer_follow_target=np.zeros((1, 29)),
    )
    with pytest.raises(ValueError, match="causal prior"):
        verify_follow_prefix(a, b, initial_control=140, receiver_velocity_controls=2)
