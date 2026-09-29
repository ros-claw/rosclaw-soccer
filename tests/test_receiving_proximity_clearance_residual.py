"""Measured-ball entry gate is causal, latched and bounded."""

from dataclasses import replace

import pytest

from rosclaw_soccer.rsi.receiving_proximity_clearance_residual import (
    ReceivingProximityClearanceResidual,
)
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactMailbox,
    ReceiveContactSnapshot,
)
from rosclaw_soccer.training.receiving_feedback import ReceivingContactHistory
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from tests.training.test_receiving_feedback import observation


def _observation(frame: int, ball_x: float):
    value = observation(frame)
    qpos = list(value.qpos)
    qpos[36] = ball_x
    qpos[37] = 0.0
    return replace(
        value,
        agent_id="red.finisher",
        qpos=tuple(qpos),
        action_substrate="A1_body29",
        previous_body_residual_rad=(0.0,) * 29,
        contact_history=ReceivingContactHistory(frame * 0.02, None),
    )


def test_proximity_entry_is_latched_from_current_ball_distance():
    mailbox = ReceiveContactMailbox("red.finisher")
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 0, 10, ((0.0,) * 29,))
    actor = ReceivingProximityClearanceResidual(
        "red.finisher",
        schedule.contract_hash,
        mailbox,
        (0.0,) * 12,
        0.03,
        0.0,
        -0.05,
        0.12,
        0.6,
    )
    mailbox._snapshot = ReceiveContactSnapshot(0.4, None, None, None, 0)
    assert actor.propose(_observation(20, 0.9)) == (0.0,) * 29
    assert actor.entry_frame is None
    mailbox._snapshot = ReceiveContactSnapshot(0.42, None, None, None, 0)
    entered = actor.propose(_observation(21, 0.5))
    assert actor.entry_frame == 21 and entered[0] == 0.03 and entered[4] == -0.05
    mailbox._snapshot = ReceiveContactSnapshot(0.44, None, None, None, 0)
    assert actor.propose(_observation(22, 0.9)) == entered
    assert actor.nonzero_frames == 2


def test_proximity_entry_rejects_invalid_threshold():
    mailbox = ReceiveContactMailbox("red.finisher")
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 0, 10, ((0.0,) * 29,))
    with pytest.raises(ValueError, match="bounded measured-ball"):
        ReceivingProximityClearanceResidual(
            "red.finisher",
            schedule.contract_hash,
            mailbox,
            (0.0,) * 12,
            0.03,
            0.0,
            -0.05,
            0.12,
            1.5,
        )
