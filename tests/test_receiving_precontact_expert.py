"""A2 is SIM_ONLY, explicitly bounded, and zero-action A1-compatible."""

from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_coordinated_feedback import ReceivingCoordinatedFeedback
from rosclaw_soccer.rsi.receiving_precontact_expert import ReceivingPrecontactExpert
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactMailbox,
    ReceiveContactSnapshot,
)
from rosclaw_soccer.training.receiving_oracle_schedule import (
    ReceivingOracleCursor,
    ReceivingOracleSchedule,
)
from tests.test_receiving_taskspace_feedback import _observation


def test_a2_zero_cursor_exactly_matches_a1_and_rejects_unbounded_feedback():
    a1 = ReceivingOracleCursor(
        ReceivingOracleSchedule("red.finisher", "A1_body29", 15, 10, ((0.0,) * 29,))
    )
    a2 = ReceivingOracleCursor(
        ReceivingOracleSchedule("red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,))
    )
    previous = np.zeros(12)
    for frame in range(40):
        before = a1.step(frame, active=True, predecessor=previous)
        after = a2.step(frame, active=True, predecessor=previous)
        assert before is None and after is None or np.array_equal(before, after)
    with pytest.raises(ValueError, match="bounded post-entry feedback"):
        a2.step(40, active=True, predecessor=previous, desired_override_rad=(0.36,) * 29)
    assert a2.faulted


def test_a2_zero_expert_keeps_coordinated_feedback_proposal():
    parent_schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))
    expert_schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 20, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox("red.finisher")
    mailbox._snapshot = ReceiveContactSnapshot(0.4, None, None, None, 0)
    parent = ReceivingCoordinatedFeedback(
        "red.finisher", parent_schedule.contract_hash, mailbox, 0.35, 0.0, 0.0
    )
    expert = ReceivingPrecontactExpert(
        "red.finisher", expert_schedule.contract_hash, mailbox, 0.35, 0.0, 0.0
    )
    observation = _observation(20)
    assert parent.propose(observation) == expert.propose(
        replace(observation, action_substrate="A2_body29_precontact")
    )
    assert expert.activation_ceiling == "SIM_ONLY"
    with pytest.raises(ValueError, match="bounded bilateral"):
        replace(expert, right_weights=(float("nan"),) * 12)


def test_a2_high_state_releases_without_entering_small_residual_history():
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 0, 10, ((0.0,) * 29,)
    )
    cursor = ReceivingOracleCursor(schedule)
    predecessor = np.zeros(12)
    for frame in range(3):
        current = cursor.step(
            frame,
            active=True,
            predecessor=predecessor,
            desired_override_rad=(0.3,) + (0.0,) * 28,
        )
        assert current is not None
    assert current[0] == pytest.approx(0.18)
    released = cursor.step(
        3,
        active=True,
        predecessor=predecessor,
        desired_override_rad=(0.0,) * 29,
    )
    assert released is not None
    assert released[0] == pytest.approx(0.12)
