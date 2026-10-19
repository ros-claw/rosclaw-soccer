import numpy as np
import pytest

from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance, ball_entry_decision


def inputs():
    q, v = np.zeros(43), np.zeros(41)
    q[2] = 0.793
    q[3] = q[39] = 1
    q[36:39] = (0.5, -0.1, 0.13)
    prior = SkillEntryAffordance((0.25, -0.1), 150, 1.0, "sha256:" + "1" * 64)
    return q, v, prior


def test_ball_position_changes_goal_and_teacher_command_not_world_state():
    q, v, prior = inputs()
    before = q.copy()
    a = ball_entry_decision(q, v, frame=30, affordance=prior)
    q[36] += 0.1
    b = ball_entry_decision(q, v, frame=30, affordance=prior)
    assert b.target_root_xy_m[0] - a.target_root_xy_m[0] == pytest.approx(0.1)
    assert b.command[0] > a.command[0] > 0
    np.testing.assert_array_equal(q[:36], before[:36])
    assert prior.contract_hash.startswith("sha256:")


def test_measured_entry_ready_does_not_require_standstill():
    q, v, prior = inputs()
    q[0] = 0.25
    v[0] = 0.2
    result = ball_entry_decision(q, v, frame=40, affordance=prior)
    assert result.ready and not result.timed_out
    assert not ball_entry_decision(q, v, frame=20, affordance=prior).ready
    v[35] = 1
    result = ball_entry_decision(q, v, frame=40, affordance=prior)
    assert result.target_root_xy_m[0] == pytest.approx(0.35)
    assert not result.ready


def test_velocity_limit_and_timeout_no_hidden_motion_extension():
    q, v, prior = inputs()
    q[36:38] = (20, -20)
    result = ball_entry_decision(q, v, frame=40, affordance=prior)
    assert np.linalg.norm(result.command[:2]) == pytest.approx(0.6)
    assert not result.ready
    late = ball_entry_decision(q, v, frame=91, affordance=prior)
    assert late.timed_out and not late.ready and late.command == (0, 0, 0)
    with pytest.raises(ValueError):
        ball_entry_decision(q, v, frame=300, affordance=prior)


def test_frozen_affordance_cannot_contain_a_mutable_offset_list():
    with pytest.raises(ValueError):
        SkillEntryAffordance([0.25, -0.1], 150, 1.0, "sha256:" + "1" * 64)
