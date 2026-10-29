import numpy as np
import pytest

from scripts.rsi_compare_motion_episodes import compare_first_episodes


def trace():
    return dict(
        initial_actor_obs=np.zeros((1, 3)),
        initial_root_pose=np.zeros((1, 7)),
        initial_root_velocity=np.zeros((1, 6)),
        initial_joint_pos=np.zeros((1, 2)),
        initial_joint_vel=np.zeros((1, 2)),
        initial_motion_id=np.array([1]),
        initial_motion_step=np.array([0]),
        valid_before=np.array([[True], [True], [False]]),
        done=np.array([[0], [1], [0]]),
        error_body_pos=np.array([[1.0], [999.0], [999.0]]),
        error_joint_pos=np.array([[2.0], [999.0], [999.0]]),
    )


def test_excludes_reset_and_terminal_state_metrics():
    a, b = trace(), trace()
    b["error_body_pos"][0, 0] = 0.5
    result = compare_first_episodes(a, b)["rows"][0]
    assert result["common_valid_post_steps"] == 1
    assert result["error_body_pos"] == {"initial_mean": 1.0, "candidate_mean": 0.5}
    assert result["initial_length"] == 2 and not result["initial_censored"]


def test_rejects_unmatched_starts_and_incorrect_masks():
    a, b = trace(), trace()
    b["initial_root_pose"][0, 0] = 1
    with pytest.raises(ValueError, match="unmatched initial"):
        compare_first_episodes(a, b)
    b = trace()
    b["valid_before"][2, 0] = True
    with pytest.raises(ValueError, match="mask contradicts"):
        compare_first_episodes(a, b)
