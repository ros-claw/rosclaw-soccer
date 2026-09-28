"""The shared-world task-space probe only changes bounded leg targets."""

from __future__ import annotations

import numpy as np

from rosclaw_soccer.skills.team.foot_kinematics import TeamFootKinematics
from rosclaw_soccer.skills.team.motor_option import (
    TeamMotorFoundation,
    TeamMotorObservation,
    TeamMotorTarget,
)
from scripts.rsi_team_taskspace_first_touch import TeamSwingMotor

ACTION = {
    "entry_frame": 30,
    "forward_cap_m": 0.08,
    "lateral_cap_m": 0.05,
    "vertical_offset_m": 0.04,
    "swing_foot_acquisition_gap_m": 0.55,
}


def _observation(frame: int, ball_x: float) -> TeamMotorObservation:
    q = np.zeros(43)
    q[3] = q[39] = 1
    q[36:39] = (ball_x, 0.0, 0.11)
    jac = np.zeros((2, 3, 6))
    jac[:, 0, 0] = 1
    jac[:, 1, 1] = 1
    jac[:, 2, 2] = 1
    agent = "red.playmaker"
    target = TeamMotorTarget((0.0,) * 29, (20.0,) * 29, (1.0,) * 29)
    return TeamMotorObservation(
        agent_id=agent,
        frame=frame,
        time_sec=frame * 0.02,
        intent="other",
        prospective_owner=False,
        qpos=tuple(float(value) for value in q),
        qvel=(0.0,) * 41,
        target_position_m=(0.0, 0.0, 0.0),
        foundation=TeamMotorFoundation(
            agent,
            frame,
            target,
            (0.0,) * 29,
            "sha256:" + "1" * 64,
            "sha256:" + "2" * 64,
        ),
        foot_kinematics=TeamFootKinematics(
            agent_id=agent,
            frame=frame,
            foot_position_world_m=((1.0, 0.0, 0.2), (1.0, -0.2, 0.1)),
            foot_linear_jacobian_world=tuple(
                tuple(tuple(float(value) for value in row) for row in side) for side in jac
            ),
            leg_joint_limits_rad=(((-1.0, 1.0),) * 6, ((-1.0, 1.0),) * 6),
        ),
    )


def test_parent_is_exact_and_candidate_is_bounded() -> None:
    parent = TeamSwingMotor("red.playmaker", False, ACTION)
    candidate = TeamSwingMotor("red.playmaker", True, ACTION)
    for frame in range(31):
        observation = _observation(frame, 2.0 if frame < 30 else 1.35)
        parent_target = np.asarray(parent.propose(observation).target_rad)
        candidate_target = np.asarray(candidate.propose(observation).target_rad)
        assert np.allclose(parent_target, 0)
        if frame < 30:
            assert np.allclose(candidate_target, 0)
    assert 0 < np.max(np.abs(candidate_target[:6])) <= 0.35
    assert np.allclose(candidate_target[6:], 0)
    assert candidate.side == 0
