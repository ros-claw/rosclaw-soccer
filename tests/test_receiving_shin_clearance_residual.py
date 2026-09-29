"""Early contact-clearance proposals stay bounded and causal."""

from dataclasses import replace

import pytest

from rosclaw_soccer.rsi.receiving_shin_clearance_residual import ReceivingShinClearanceResidual
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactMailbox,
    ReceiveContactSnapshot,
)
from rosclaw_soccer.training.receiving_feedback import ReceivingContactHistory
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from tests.training.test_receiving_feedback import observation


def _observation(frame: int, *, ball_y: float):
    value = observation(frame)
    qpos = list(value.qpos)
    qpos[37] = ball_y
    return replace(
        value,
        agent_id="red.finisher",
        qpos=tuple(qpos),
        action_substrate="A1_body29",
        previous_body_residual_rad=(0.0,) * 29,
        contact_history=ReceivingContactHistory(frame * 0.02, None),
    )


def test_precontact_right_foot_clearance_and_postcontact_expiry():
    mailbox = ReceiveContactMailbox("red.finisher")
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 0, 10, ((0.0,) * 29,))
    actor = ReceivingShinClearanceResidual(
        "red.finisher", schedule.contract_hash, mailbox, (0.0,) * 12, 0.05, 0.0, -0.1
    )
    mailbox._snapshot = ReceiveContactSnapshot(0.4, None, None, None, 0)
    before = actor.propose(_observation(20, ball_y=-0.08))
    assert before[6] == 0.05 and before[10] == -0.1
    assert sum(x != 0 for x in before) == 2
    mailbox._snapshot = ReceiveContactSnapshot(0.42, 0.416, "right_foot", 0.0, 0)
    after = actor.propose(_observation(21, ball_y=0.08))
    assert after == before  # Measured foot supersedes the later ball side.
    mailbox._snapshot = ReceiveContactSnapshot(0.44, 0.3, "right_foot", 0.0, 0)
    assert actor.propose(_observation(22, ball_y=0.08)) == (0.0,) * 29
    assert actor.nonzero_frames == 2
    assert not hasattr(actor, "motor_target")


def test_clearance_rejects_unbounded_or_mixed_parent():
    mailbox = ReceiveContactMailbox("red.finisher")
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 0, 10, ((0.0,) * 29,))
    with pytest.raises(ValueError, match="bounded clearance-only"):
        ReceivingShinClearanceResidual(
            "red.finisher", schedule.contract_hash, mailbox, (0.0,) * 12, 0.11, 0.0, 0.0
        )
    with pytest.raises(ValueError, match="bounded clearance-only"):
        ReceivingShinClearanceResidual(
            "red.finisher", schedule.contract_hash, mailbox, (0.1,) * 12, 0.0, 0.0, 0.0
        )
