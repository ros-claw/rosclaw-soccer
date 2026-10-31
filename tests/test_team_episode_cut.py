import numpy as np
import pytest

from rosclaw_soccer.training.team_episode_cut import sampled_team_episode_cut


def trace():
    n = 5
    pose = np.tile(np.array([0, 0, 0.75, 1, 0, 0, 0], dtype=float), (n, 1))
    return {
        "time": np.arange(n) * 0.02,
        "ball_pose": np.tile(np.array([0, 0, 0.11, 1, 0, 0, 0], dtype=float), (n, 1)),
        "pitch_boundary_geometry": np.tile(np.array([-1.5, 7.5, 3, 0.11, 2.6, 1.8]), (n, 1)),
        "robot_robot_contact_count": np.zeros(n),
        "blue_finisher_pelvis_pose": pose.copy(),
        "blue_finisher_joint_safety_margin_rad": np.ones((n, 29)),
        "red_finisher_pelvis_pose": pose.copy(),
        "red_finisher_joint_safety_margin_rad": np.ones((n, 29)),
    }


IDS = ("blue.finisher", "red.finisher")


def test_clean_and_sampled_only():
    cut = sampled_team_episode_cut(trace(), agent_ids=IDS)
    assert cut.first_terminal_frame is None
    assert cut.sampled_only and not cut.promotion_eligible


def test_all_agents_and_exit_use_first_event():
    data = trace()
    data["blue_finisher_joint_safety_margin_rad"][2, 3] = -0.01
    data["ball_pose"][3, 1] = 3.2
    cut = sampled_team_episode_cut(data, agent_ids=IDS)
    assert (cut.first_safety_frame, cut.first_exit_frame, cut.first_terminal_frame) == (2, 3, 2)


@pytest.mark.parametrize("failure", ["collision", "height", "tilt"])
def test_body_or_collision_failure(failure):
    data = trace()
    if failure == "collision":
        data["robot_robot_contact_count"][1] = 1
    elif failure == "height":
        data["red_finisher_pelvis_pose"][1, 2] = 0.54
    else:
        data["red_finisher_pelvis_pose"][1, 3:7] = [np.cos(0.5), np.sin(0.5), 0, 0]
    assert sampled_team_episode_cut(data, agent_ids=IDS).first_safety_frame == 1


def test_rejects_bad_inputs():
    data = trace()
    data["red_finisher_pelvis_pose"][2, 3] = np.nan
    with pytest.raises(ValueError):
        sampled_team_episode_cut(data, agent_ids=IDS)
    with pytest.raises(ValueError):
        sampled_team_episode_cut(trace(), agent_ids=("red.finisher", "blue.finisher"))
