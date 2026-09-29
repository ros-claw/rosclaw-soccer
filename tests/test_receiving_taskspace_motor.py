"""Exclusive receiving motor preserves foundation and bounds correction."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_taskspace_motor import ReceivingTaskspaceMotor
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactMailbox,
    ReceiveContactSnapshot,
)
from rosclaw_soccer.skills.team.foot_kinematics import TeamFootKinematics
from rosclaw_soccer.skills.team.motor_option import (
    TeamMotorFoundation,
    TeamMotorObservation,
    TeamMotorTarget,
)


def _observation(frame: int) -> TeamMotorObservation:
    q = np.zeros(43)
    q[2] = 0.75
    q[3] = q[39] = 1.0
    q[36:39] = (0.2, 0.08, 0.115)
    jacobian = (
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    )
    agent = "red.finisher"
    target = TeamMotorTarget((0.0,) * 29, (20.0,) * 29, (1.0,) * 29)
    return TeamMotorObservation(
        agent_id=agent,
        frame=frame,
        time_sec=frame * 0.02,
        intent="other",
        prospective_owner=False,
        qpos=tuple(float(x) for x in q),
        qvel=(0.0,) * 41,
        target_position_m=(0.0, 0.0, 0.0),
        foundation=TeamMotorFoundation(
            agent, frame, target, (0.0,) * 29, "sha256:" + "1" * 64, "sha256:" + "2" * 64
        ),
        foot_kinematics=TeamFootKinematics(
            agent,
            frame,
            ((0.0, 0.0, 0.06),) * 2,
            (jacobian,) * 2,
            (((-1.0, 1.0),) * 6,) * 2,
            ((0.0, 0.0, 0.0),) * 2,
        ),
    )


def test_zero_motor_is_exact_parent_and_nonzero_is_bounded():
    mailbox = ReceiveContactMailbox("red.finisher")
    mailbox._snapshot = ReceiveContactSnapshot(0.0, None, None, None, 0)
    zero = ReceivingTaskspaceMotor("red.finisher", mailbox, 0.0, 0.0, 0.0)
    candidate = ReceivingTaskspaceMotor("red.finisher", mailbox, 0.5, 0.5, 0.05)
    value = _observation(0)
    assert zero.propose(value) is value.foundation.target
    target = candidate.propose(value)
    assert max(abs(x) for x in target.target_rad[:6]) <= 0.25
    assert any(target.target_rad[:6]) and target.target_rad[6:] == (0.0,) * 23
    assert candidate.nonzero_target_frames == 1
    assert candidate.peak_correction_rad <= 0.25
    assert candidate.activation_ceiling == "SIM_ONLY"
    with pytest.raises(ValueError, match="consecutive same-player"):
        candidate.propose(value)


def test_motor_rejects_stale_contact_and_unbounded_gain():
    mailbox = ReceiveContactMailbox("red.finisher")
    with pytest.raises(ValueError, match="bounded same-player"):
        ReceivingTaskspaceMotor("red.finisher", mailbox, 1.1, 0.0, 0.0)
    with pytest.raises(ValueError, match="bounded same-player"):
        ReceivingTaskspaceMotor("red.finisher", mailbox, 0.0, 0.0, 0.0, target_depth_m=0.02)
    with pytest.raises(ValueError, match="bounded same-player"):
        ReceivingTaskspaceMotor("red.finisher", mailbox, 0.0, 0.0, 0.0, target_lateral_m=0.5)
    actor = ReceivingTaskspaceMotor("red.finisher", mailbox, 0.5, 0.0, 0.0)
    with pytest.raises(ValueError, match="consecutive same-player"):
        actor.propose(_observation(1))


def test_idle_motor_yields_until_completed_own_foot_touch():
    mailbox = ReceiveContactMailbox("red.finisher")
    actor = ReceivingTaskspaceMotor(
        "red.finisher", mailbox, 0.0, 0.5, 0.05, idle_before_first_touch=True
    )
    assert actor.propose(_observation(0)) is None
    mailbox._snapshot = ReceiveContactSnapshot(0.02, 0.018, "left_foot", 0.0, 0)
    target = actor.propose(_observation(1))
    assert target is not None and any(target.target_rad[:6])
    assert actor.nonzero_target_frames == 1
