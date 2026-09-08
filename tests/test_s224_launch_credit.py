import numpy as np
import pytest

from rosclaw_soccer.training.near_ball_residual_ppo import physical_rewards


def delayed_pass_trace():
    ids = tuple(
        f"{team}.{role}"
        for team in ("blue", "red")
        for role in ("defender", "finisher", "goalkeeper", "playmaker")
    )
    n = 61
    trace = {
        "time": np.arange(n) * 0.1,
        "residual_observations": np.zeros((n, 8, 56)),
        "residual_applied": np.zeros((n, 8, 12)),
        "pass_source_agent_code": np.zeros(n),
        "pass_target_agent_code": np.zeros(n),
        "ball_contact_agent_code": np.zeros(n),
        "ball_contact_effector_code": np.ones(n),
        "ball_contact_force_n": np.ones(n),
        "ball_nonfoot_contact_agent_code": np.zeros(n),
        "ball_nonfoot_contact_force_n": np.zeros(n),
        "robot_robot_contact_first_code": np.zeros(n),
        "robot_robot_contact_second_code": np.zeros(n),
        "ball_pose": np.zeros((n, 7)),
        "ball_velocity": np.zeros((n, 6)),
    }
    for agent in ids:
        key = agent.replace(".", "_")
        trace[key + "_left_foot_position"] = np.tile([1.0, 0, 0], (n, 1))
        trace[key + "_right_foot_position"] = np.tile([1.0, 0.1, 0], (n, 1))
        trace[key + "_target_position"] = np.tile([1.0, 0, 0], (n, 1))
        trace[key + "_pelvis_pose"] = np.tile([0, 0, 0.78], (n, 1))
    trace["pass_source_agent_code"][:25] = 8
    trace["pass_target_agent_code"][:25] = 6
    trace["ball_contact_agent_code"][20] = 8
    trace["ball_contact_agent_code"][45] = 6
    trace["ball_pose"][:, 0] = np.clip((np.arange(n) - 20) / 25, 0, 1)
    trace["ball_velocity"][:, 0] = 0.4
    return ids, trace


def test_launch_credit_reaches_exactly_both_participants_after_setup_delay():
    ids, trace = delayed_pass_trace()
    legacy = physical_rewards(trace, ids)
    trace["pass_feedback_launch_relative"] = np.ones(len(trace["time"]), dtype=bool)
    difference = physical_rewards(trace, ids) - legacy
    expected = np.zeros((61, 8))
    expected[45, [5, 7]] = 1
    np.testing.assert_allclose(difference, expected, atol=1e-12)


def test_nonfoot_interruption_and_invalid_contract_cannot_earn_completion():
    ids, trace = delayed_pass_trace()
    trace["ball_nonfoot_contact_agent_code"][30] = 8
    trace["ball_nonfoot_contact_force_n"][30] = 5
    legacy = physical_rewards(trace, ids)
    trace["pass_feedback_launch_relative"] = np.ones(61, dtype=bool)
    np.testing.assert_array_equal(physical_rewards(trace, ids), legacy)
    trace["pass_feedback_launch_relative"] = np.ones(61, dtype=float)
    with pytest.raises(ValueError, match="contract"):
        physical_rewards(trace, ids)
