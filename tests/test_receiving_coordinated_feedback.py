"""Mirrored whole-body receiving proposals stay inside the single A1 guard."""

from dataclasses import replace

import pytest

from rosclaw_soccer.rsi.receiving_coordinated_feedback import ReceivingCoordinatedFeedback
from rosclaw_soccer.rsi.receiving_taskspace_feedback import ReceivingTaskspaceFeedback
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactMailbox,
    ReceiveContactSnapshot,
)
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from tests.test_receiving_taskspace_feedback import _observation


def test_zero_synergies_match_parent_and_active_synergies_are_bounded():
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))
    mailbox = ReceiveContactMailbox("red.finisher")
    mailbox._snapshot = ReceiveContactSnapshot(0.4, None, None, None, 0)
    parent = ReceivingTaskspaceFeedback(
        "red.finisher", schedule.contract_hash, mailbox, 0.35, 0.0, 0.0
    )
    zero = ReceivingCoordinatedFeedback(
        "red.finisher", schedule.contract_hash, mailbox, 0.35, 0.0, 0.0
    )
    actor = ReceivingCoordinatedFeedback(
        "red.finisher",
        schedule.contract_hash,
        mailbox,
        0.35,
        0.0,
        0.0,
        coordination=(1.0, 0.5, -0.5, 0.25, 0.0, 0.0, 0.0, 0.0),
    )
    current = _observation(20)
    parent_action = parent.propose(current)
    assert zero.propose(current) == parent_action
    action = actor.propose(current)
    assert action != parent_action
    assert any(action[6:12]) and any(action[12:])
    assert max(map(abs, action)) <= 0.1
    assert not hasattr(actor, "motor_target")


def test_right_touch_selects_post_contact_mirrored_synergies():
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))
    mailbox = ReceiveContactMailbox("red.finisher")
    mailbox._snapshot = ReceiveContactSnapshot(0.4, 0.39, "right_foot", 0.0, 0)
    actor = ReceivingCoordinatedFeedback(
        "red.finisher",
        schedule.contract_hash,
        mailbox,
        coordination=(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    )
    action = actor.propose(_observation(20))
    assert action[9] > 0 and action[10] < 0
    assert action[:6] == (0.0,) * 6
    with pytest.raises(ValueError, match="eight bounded"):
        replace(actor, coordination=(float("nan"),) * 8)


def test_coordinated_action_exits_after_contact_window():
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))
    mailbox = ReceiveContactMailbox("red.finisher")
    mailbox._snapshot = ReceiveContactSnapshot(1.62, 0.4, "right_foot", 0.0, 0)
    actor = ReceivingCoordinatedFeedback(
        "red.finisher",
        schedule.contract_hash,
        mailbox,
        coordination=(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
    )
    assert actor.propose(_observation(81)) == (0.0,) * 29
