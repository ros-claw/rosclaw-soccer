import numpy as np
import pytest

from rosclaw_soccer.training.in_play_rewards import in_play_rewards

IDS = tuple(f"{team}.{role}" for team in ("blue", "red") for role in ("a", "b", "c", "d"))


def fixture():
    return {
        "time": np.arange(4) * 0.02,
        "ball_pose": np.array(
            [[3.0, 0.0, 0.12], [3.0, 0.0, 0.12], [3.0, 3.2, 0.12], [3.0, 0.0, 0.12]]
        ),
        "pitch_boundary_geometry": np.tile([-1.5, 7.5, 3.0, 0.115, 3.0, 2.0], (4, 1)),
        "ball_contact_agent_code": np.array([5, 0, 0, 0]),
        "ball_contact_force_n": np.array([10.0, 0, 0, 0]),
        "ball_nonfoot_contact_agent_code": np.zeros(4),
        "ball_nonfoot_contact_force_n": np.zeros(4),
    }


def test_first_out_cuts_later_credit_without_resetting_or_mutating_physics():
    trace = fixture()
    before = {k: v.copy() for k, v in trace.items()}
    rewards = np.ones((4, 8))
    result = in_play_rewards(rewards, trace, IDS)
    np.testing.assert_array_equal(result[:2], rewards[:2])
    np.testing.assert_array_equal(result[2], [0.0, 0.0, 0.0, 0.0, -0.25, -0.25, -0.25, -0.25])
    assert not result[3].any()
    for key in trace:
        np.testing.assert_array_equal(trace[key], before[key])
    assert (rewards == 1).all()


def test_goal_scores_correct_team_once_and_keeps_terminal_safety_cost():
    trace = fixture()
    trace["ball_pose"][2:] = [[7.7, 0, 1.0], [-1.7, 0, 1.0]]
    rewards = np.ones((4, 8))
    rewards[2, 4] = -0.2
    result = in_play_rewards(rewards, trace, IDS)
    np.testing.assert_allclose(result[2], [-0.5, -0.5, -0.5, -0.5, 0.3, 0.5, 0.5, 0.5])
    assert not result[3].any()


def test_ambiguous_same_frame_touch_does_not_invent_an_individual_owner():
    trace = fixture()
    trace["ball_nonfoot_contact_agent_code"][0] = 2
    trace["ball_nonfoot_contact_force_n"][0] = 5
    assert not in_play_rewards(np.zeros((4, 8)), trace, IDS).any()


@pytest.mark.parametrize("damage", ["missing", "moving", "nan", "unknown_actor", "initial_out"])
def test_bad_or_missing_geometry_cannot_be_used_for_pitch_credit(damage):
    trace = fixture()
    if damage == "missing":
        del trace["pitch_boundary_geometry"]
    elif damage == "moving":
        trace["pitch_boundary_geometry"][1, 1] = 9
    elif damage == "nan":
        trace["ball_pose"][2, 0] = np.nan
    elif damage == "initial_out":
        trace["ball_pose"][0, 0] = 9
    else:
        trace["ball_contact_agent_code"][0] = 9
    with pytest.raises(ValueError):
        in_play_rewards(np.zeros((4, 8)), trace, IDS)
