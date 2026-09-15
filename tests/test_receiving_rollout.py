import numpy as np
import pytest

from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window


def trace():
    n = 101
    ball = np.zeros((n, 7))
    ball[:, 2] = 0.115
    ball[:, 3] = 1
    body = np.zeros((n, 7))
    body[:, 2] = 0.75
    body[:, 3] = 1
    return {
        "time": np.arange(1, n + 1) * 0.02,
        "ball_pose": ball,
        "ball_velocity": np.zeros((n, 6)),
        "red_receiver_left_foot_position": ball[:, :3] + np.array([0.15, 0, 0]),
        "red_receiver_right_foot_position": ball[:, :3] + np.array([0.2, 0, 0]),
        "ball_contact_agent_code": np.zeros(n),
        "ball_contact_effector_code": np.zeros(n),
        "ball_contact_force_n": np.zeros(n),
        "ball_nonfoot_contact_agent_code": np.zeros(n),
        "ball_nonfoot_contact_force_n": np.zeros(n),
        "red_receiver_pelvis_pose": body,
        "red_receiver_joint_safety_margin_rad": np.full((n, 29), 0.1),
        "robot_robot_contact_count": np.zeros(n),
    }


def assess(t):
    return receiving_window(
        t, agent_ids=("blue.other", "red.receiver"), agent_id="red.receiver", start=1, frames=100
    )


def contact(t, frame=20):
    t["ball_contact_agent_code"][frame] = 2
    t["ball_contact_effector_code"][frame] = 1
    t["ball_contact_force_n"][frame] = 2


def test_near_stationary_ball_without_foot_never_counts():
    r, d = assess(trace())
    assert not d["controlled_reception"] and d["first_foot_contact_sec"] is None
    assert r[-1] == -2


def test_actual_touch_then_half_second_observation_and_stable_tail():
    t = trace()
    contact(t)
    r, d = assess(t)
    assert d["controlled_reception"] and r[-1] > 9
    before = {k: v.copy() for k, v in t.items()}
    assess(t)
    assert all(np.array_equal(t[k], v) for k, v in before.items())


@pytest.mark.parametrize("fault", ["late", "body", "other", "joint", "collision", "fast"])
def test_bad_receiving_windows_never_get_control_bonus(fault):
    t = trace()
    contact(t, 99 if fault == "late" else 20)
    if fault == "body":
        t["ball_nonfoot_contact_agent_code"][25] = 2
        t["ball_nonfoot_contact_force_n"][25] = 1
    if fault == "other":
        t["ball_contact_agent_code"][25] = 1
        t["ball_contact_force_n"][25] = 1
    if fault == "joint":
        t["red_receiver_joint_safety_margin_rad"][25, 0] = -0.01
    if fault == "collision":
        t["robot_robot_contact_count"][25] = 1
    if fault == "fast":
        t["ball_velocity"][-5:, 0] = 1
    assert not assess(t)[1]["controlled_reception"]


def test_bad_time_and_foreign_code_fail_closed():
    t = trace()
    t["time"][2] = t["time"][1]
    with pytest.raises(ValueError):
        assess(t)
    t = trace()
    t["ball_contact_agent_code"][3] = 4
    with pytest.raises(ValueError):
        assess(t)


def explain(t, frames=100):
    return explain_receiving_window(
        t,
        agent_ids=("blue.other", "red.receiver"),
        agent_id="red.receiver",
        start=1,
        frames=frames,
    )


@pytest.mark.parametrize(
    "fault,criterion",
    [
        ("missing", "measured_own_foot_contact"),
        ("late", "half_second_after_contact"),
        ("fast", "slow_ball_tail"),
        ("far", "near_foot_tail"),
        ("high", "low_ball_tail"),
        ("foreign", "no_forbidden_contact_after_touch"),
        ("unsafe", "body_and_collision_safe"),
        ("short", "complete_window"),
    ],
)
def test_explanation_identifies_independent_failure(fault, criterion):
    t = trace()
    if fault != "missing":
        contact(t, 99 if fault == "late" else 20)
    if fault == "fast":
        t["ball_velocity"][-5, 0] = 0.36  # Final speed alone would miss this.
    if fault == "far":
        t["red_receiver_left_foot_position"][-5, 0] = 0.36
        t["red_receiver_right_foot_position"][-5, 0] = 0.36
    if fault == "high":
        t["ball_pose"][-5, 2] = 0.21
    if fault == "foreign":
        t["ball_nonfoot_contact_agent_code"][25] = 1
        t["ball_nonfoot_contact_force_n"][25] = 1
    if fault == "unsafe":
        t["red_receiver_joint_safety_margin_rad"][25, 0] = -0.01
    before = {k: v.copy() for k, v in t.items()}
    result = explain(t, 99 if fault == "short" else 100)
    assert criterion in result["failed_criteria"]
    assert not result["controlled_reception"]
    assert all(np.array_equal(t[k], v) for k, v in before.items())


def test_explanation_retains_success_and_ignores_pre_touch_foreign_contact():
    t = trace()
    contact(t)
    t["ball_nonfoot_contact_agent_code"][10] = 1
    t["ball_nonfoot_contact_force_n"][10] = 1
    result = explain(t)
    assert result["controlled_reception"] and not result["failed_criteria"]
    assert result["forbidden_post_contact_frames"] == 0
    assert result["activation_ceiling"] == "SIM_ONLY"
    assert not result["promotion_eligible"]


def test_explanation_does_not_swallow_invalid_evidence():
    t = trace()
    t["ball_velocity"][0, 0] = np.nan
    with pytest.raises(ValueError):
        explain(t)
    t = trace()
    t["ball_contact_agent_code"][3] = 4
    with pytest.raises(ValueError):
        explain(t)
