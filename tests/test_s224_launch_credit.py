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


@pytest.mark.parametrize("shaping", ["legacy", "terminal_potential_v1"])
def test_launch_credit_reaches_exactly_both_participants_after_setup_delay(shaping):
    ids, trace = delayed_pass_trace()
    legacy = physical_rewards(trace, ids, reward_shaping=shaping, gamma=0.997)
    trace["pass_feedback_launch_relative"] = np.ones(len(trace["time"]), dtype=bool)
    difference = physical_rewards(trace, ids, reward_shaping=shaping, gamma=0.997) - legacy
    expected = np.zeros((61, 8))
    expected[45, [5, 7]] = 1
    np.testing.assert_allclose(difference, expected, atol=1e-12)


@pytest.mark.parametrize("shaping", ["legacy", "terminal_potential_v1"])
def test_nonfoot_interruption_and_invalid_contract_cannot_earn_completion(shaping):
    ids, trace = delayed_pass_trace()
    trace["ball_nonfoot_contact_agent_code"][30] = 8
    trace["ball_nonfoot_contact_force_n"][30] = 5
    legacy = physical_rewards(trace, ids, reward_shaping=shaping, gamma=0.997)
    trace["pass_feedback_launch_relative"] = np.ones(61, dtype=bool)
    np.testing.assert_array_equal(
        physical_rewards(trace, ids, reward_shaping=shaping, gamma=0.997), legacy
    )
    trace["pass_feedback_launch_relative"] = np.ones(61, dtype=float)
    with pytest.raises(ValueError, match="contract"):
        physical_rewards(trace, ids, reward_shaping=shaping, gamma=0.997)


def test_motor_task_reward_changes_only_contact_direction_and_preserves_guards():
    ids, trace = delayed_pass_trace()
    for agent in ids:
        trace[agent.replace(".", "_") + "_joint_safety_margin_rad"] = np.ones((61, 29))
    trace["option_agent_code"] = np.zeros(61, dtype=np.int64)
    trace["option_agent_code"][45] = 6
    trace["option_target_position_m"] = np.tile([7.5, 1, 0.115], (61, 1))
    trace["red_finisher_target_position"][45] = [-1, 0, 0]
    trace["ball_velocity"][45, :2] = [8, 0]
    old = physical_rewards(trace, ids, reward_shaping="contact_safety_v1", gamma=0.997)
    bound = physical_rewards(trace, ids, reward_shaping="motor_task_contact_v1", gamma=0.997)
    difference = np.zeros((61, 8))
    difference[45, 5] = 0.16  # -0.08 wrong-way credit becomes +0.08.
    np.testing.assert_allclose(bound - old, difference, atol=1e-12)
    del trace["option_target_position_m"]
    with pytest.raises(ValueError, match="recorded option targets"):
        physical_rewards(trace, ids, reward_shaping="motor_task_contact_v1", gamma=0.997)
    np.testing.assert_array_equal(
        physical_rewards(trace, ids, reward_shaping="contact_safety_v1", gamma=0.997), old
    )


def test_parallel_motor_rewards_use_each_body_target_not_the_empty_scalar_slot():
    ids, trace = delayed_pass_trace()
    trace["option_agent_code"] = np.zeros(61, dtype=np.int64)
    trace["option_target_position_m"] = np.zeros((61, 3))
    for agent in ids:
        key = agent.replace(".", "_")
        trace[key + "_joint_safety_margin_rad"] = np.ones((61, 29))
        trace[key + "_motor_option_active"] = np.zeros(61, dtype=bool)
        trace[key + "_motor_option_target_m"] = np.zeros((61, 3))
    for agent, direction in (("red.finisher", 1), ("blue.finisher", -1)):
        key = agent.replace(".", "_")
        trace[key + "_motor_option_active"][44:46] = True
        trace[key + "_motor_option_target_m"][44:46] = [direction * 7.5, 0, 1.5]
        trace[key + "_target_position"][44:46] = [-direction * 7.5, 0, 1.5]
    trace["ball_contact_agent_code"][44] = 2
    trace["ball_velocity"][44, 0] = -8
    trace["ball_velocity"][45, 0] = 8
    old = physical_rewards(trace, ids, reward_shaping="motor_task_contact_v1", gamma=0.997)
    trace["per_player_motor_contract"] = np.ones(61, dtype=bool)
    bound = physical_rewards(trace, ids, reward_shaping="motor_task_contact_v1", gamma=0.997)
    expected = np.zeros((61, 8))
    expected[44, 1] = 0.16
    expected[45, 5] = 0.16
    np.testing.assert_allclose(bound - old, expected, atol=1e-12)
