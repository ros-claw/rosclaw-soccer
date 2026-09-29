"""Whole-body search proposals stay typed, bounded, and contact-causal."""

from dataclasses import replace

import pytest

from rosclaw_soccer.rsi.receiving_whole_body_residual import ReceivingWholeBodyResidual
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactMailbox,
    ReceiveContactSnapshot,
)
from rosclaw_soccer.training.receiving_feedback import ReceivingContactHistory
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from tests.training.test_receiving_feedback import observation


def _observation(frame: int):
    return replace(
        observation(frame),
        agent_id="red.finisher",
        action_substrate="A1_body29",
        previous_body_residual_rad=(0.0,) * 29,
        contact_history=ReceivingContactHistory(frame * 0.02, None),
    )


def test_contact_switch_changes_bounded_whole_body_proposal():
    mailbox = ReceiveContactMailbox("red.finisher")
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))
    actor = ReceivingWholeBodyResidual(
        "red.finisher", schedule.contract_hash, mailbox, (0.5,) * 6 + (-0.5,) * 6
    )
    mailbox._snapshot = ReceiveContactSnapshot(0.4, None, None, None, 0)
    before = actor.propose(_observation(20))
    mailbox._snapshot = ReceiveContactSnapshot(0.42, 0.416, "left_foot", 0.0, 0)
    after = actor.propose(_observation(21))
    assert len(before) == len(after) == 29
    assert max(abs(x) for x in before + after) <= 0.1
    assert after == tuple(-x for x in before)
    assert actor.nonzero_frames == 2
    assert not hasattr(actor, "motor_target") and not hasattr(actor, "observe_physics")
    with pytest.raises(ValueError, match="consecutive completed"):
        actor.propose(_observation(21))


@pytest.mark.parametrize("coefficients", [(0.0,) * 11, (float("nan"),) + (0.0,) * 11])
def test_invalid_coefficients_fail_closed(coefficients):
    mailbox = ReceiveContactMailbox("red.finisher")
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))
    with pytest.raises(ValueError, match="bounded same-player"):
        ReceivingWholeBodyResidual("red.finisher", schedule.contract_hash, mailbox, coefficients)
