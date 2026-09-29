"""Pre-impact orientation proposals are mirrored, bounded, and contact-gated."""

from dataclasses import replace

import pytest

from rosclaw_soccer.rsi.receiving_coordinated_feedback import ReceivingCoordinatedFeedback
from rosclaw_soccer.rsi.receiving_impact_orientation_feedback import (
    ReceivingImpactOrientationFeedback,
)
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactMailbox,
    ReceiveContactSnapshot,
)
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from tests.test_receiving_taskspace_feedback import _observation


def test_zero_orientation_is_identity_and_hip_rotation_mirrors():
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))
    mailbox = ReceiveContactMailbox("red.finisher")
    mailbox._snapshot = ReceiveContactSnapshot(0.4, None, None, None, 0)
    parent = ReceivingCoordinatedFeedback("red.finisher", schedule.contract_hash, mailbox)
    zero = ReceivingImpactOrientationFeedback("red.finisher", schedule.contract_hash, mailbox)
    actor = ReceivingImpactOrientationFeedback(
        "red.finisher",
        schedule.contract_hash,
        mailbox,
        impact_roll=1.0,
        impact_yaw=-1.0,
    )
    current = _observation(20)
    assert zero.propose(current) == parent.propose(current)
    action = actor.propose(current)
    assert action[1] > 0 and action[2] < 0 and action[5] < 0
    assert max(map(abs, action)) <= 0.1
    right_q = list(current.qpos)
    right_q[37] = -0.1
    right = replace(current, qpos=tuple(right_q))
    right_actor = ReceivingImpactOrientationFeedback(
        "red.finisher",
        schedule.contract_hash,
        mailbox,
        impact_roll=1.0,
        impact_yaw=-1.0,
    )
    mirrored = right_actor.propose(right)
    assert mirrored[7] < 0 and mirrored[8] > 0 and mirrored[11] > 0


def test_orientation_stops_after_measured_foot_contact():
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))
    mailbox = ReceiveContactMailbox("red.finisher")
    mailbox._snapshot = ReceiveContactSnapshot(0.4, 0.39, "left_foot", 0.0, 0)
    actor = ReceivingImpactOrientationFeedback(
        "red.finisher",
        schedule.contract_hash,
        mailbox,
        impact_roll=1.0,
        impact_yaw=1.0,
    )
    assert actor.propose(_observation(20)) == (0.0,) * 29
    with pytest.raises(ValueError, match="bounded finite"):
        replace(actor, impact_roll=float("inf"))
