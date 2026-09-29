"""Task-space A1 proposals are current-foot-only and bounded."""

from dataclasses import replace

import pytest

from rosclaw_soccer.rsi.receiving_taskspace_feedback import ReceivingTaskspaceFeedback
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactMailbox,
    ReceiveContactSnapshot,
)
from rosclaw_soccer.skills.team.foot_kinematics import TeamFootKinematics
from rosclaw_soccer.skills.team.shin_clearance import TeamShinClearance
from rosclaw_soccer.training.receiving_feedback import ReceivingContactHistory
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from tests.training.test_receiving_feedback import observation


def _feet(frame: int):
    jacobian = (
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    )
    return TeamFootKinematics(
        "red.finisher",
        frame,
        ((0.0, 0.0, 0.06),) * 2,
        (jacobian,) * 2,
        (((-1.0, 1.0),) * 6,) * 2,
        ((0.0, 0.0, 0.0),) * 2,
    )


def _observation(frame: int):
    value = observation(frame)
    q = list(value.qpos)
    q[36:39] = (0.2, 0.1, 0.115)
    return replace(
        value,
        agent_id="red.finisher",
        qpos=tuple(q),
        action_substrate="A1_body29",
        previous_body_residual_rad=(0.0,) * 29,
        contact_history=ReceivingContactHistory(frame * 0.02, None),
        foot_kinematics=_feet(frame),
    )


def test_taskspace_uses_current_foot_and_zero_is_identity():
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))
    mailbox = ReceiveContactMailbox("red.finisher")
    mailbox._snapshot = ReceiveContactSnapshot(0.4, None, None, None, 0)
    zero = ReceivingTaskspaceFeedback(
        "red.finisher", schedule.contract_hash, mailbox, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    assert zero.propose(_observation(20)) == (0.0,) * 29
    actor = ReceivingTaskspaceFeedback(
        "red.finisher", schedule.contract_hash, mailbox, 0.5, 0.5, 0.05, 0.0, 0.0
    )
    action = actor.propose(_observation(20))
    assert any(action[:6]) and action[6:] == (0.0,) * 23
    assert max(abs(x) for x in action) <= 0.1
    mailbox._snapshot = ReceiveContactSnapshot(0.42, 0.416, "right_foot", 0.0, 0)
    after = actor.propose(_observation(21))
    assert after[:6] == (0.0,) * 6 and any(after[6:12])
    assert actor.nonzero_frames == 2
    assert not hasattr(actor, "motor_target")


def test_taskspace_requires_finite_bounded_gains():
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))
    mailbox = ReceiveContactMailbox("red.finisher")
    with pytest.raises(ValueError, match="bounded task-space"):
        ReceivingTaskspaceFeedback(
            "red.finisher", schedule.contract_hash, mailbox, 1.5, 0.0, 0.0, 0.0, 0.0
        )
    with pytest.raises(ValueError, match="bounded task-space"):
        ReceivingTaskspaceFeedback(
            "red.finisher", schedule.contract_hash, mailbox, target_depth_m=0.01
        )
    with pytest.raises(ValueError, match="bounded task-space"):
        ReceivingTaskspaceFeedback(
            "red.finisher", schedule.contract_hash, mailbox, target_lateral_m=0.5
        )


def test_shin_clearance_observation_is_same_player_and_frame():
    baseline = _observation(20)
    shin = TeamShinClearance("red.finisher", 20, (0.01, 0.02), ((0.0,) * 6,) * 2)
    assert replace(baseline, shin_clearance=shin).observation_hash != baseline.observation_hash
    with pytest.raises(ValueError, match="same-player current shin"):
        replace(baseline, shin_clearance=replace(shin, agent_id="blue.finisher"))
    with pytest.raises(ValueError, match="same-player current shin"):
        replace(baseline, shin_clearance=replace(shin, frame=19))
