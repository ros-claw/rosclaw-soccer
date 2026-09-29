"""One guarded A1 feedback owner trades shin clearance against foot displacement."""

from dataclasses import replace

import pytest

from rosclaw_soccer.rsi.receiving_shin_clearance_feedback import ReceivingShinClearanceFeedback
from rosclaw_soccer.rsi.receiving_taskspace_feedback import ReceivingTaskspaceFeedback
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactMailbox,
    ReceiveContactSnapshot,
)
from rosclaw_soccer.skills.team.shin_clearance import TeamShinClearance
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from tests.test_receiving_taskspace_feedback import _observation


def test_zero_clearance_gain_is_exact_foot_feedback_and_nonzero_is_bounded():
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))
    mailbox = ReceiveContactMailbox("red.finisher")
    mailbox._snapshot = ReceiveContactSnapshot(0.4, None, None, None, 0)
    qpos = list(_observation(20).qpos)
    qpos[2] = 0.75
    observation = replace(
        _observation(20),
        qpos=tuple(qpos),
        shin_clearance=TeamShinClearance(
            "red.finisher", 20, (0.01, 0.2), ((0.1, 0.0, 0.0, 0.2, 0.0, 0.0), (0.0,) * 6)
        ),
    )
    parent = ReceivingTaskspaceFeedback(
        "red.finisher", schedule.contract_hash, mailbox, 0.25, 0.0, 0.0
    )
    zero = ReceivingShinClearanceFeedback(
        "red.finisher", schedule.contract_hash, mailbox, 0.25, 0.0, 0.0
    )
    active = ReceivingShinClearanceFeedback(
        "red.finisher",
        schedule.contract_hash,
        mailbox,
        0.25,
        0.0,
        0.0,
        clearance_gain=1.0,
        target_clearance_m=0.06,
        foot_preservation=1.0,
    )
    baseline = parent.propose(observation)
    assert zero.propose(observation) == baseline
    target = active.propose(observation)
    assert target != baseline
    assert max(abs(x) for x in target) <= 0.1
    assert active.clearance_action_frames == 1
    assert active.peak_predicted_foot_shift_m <= 0.015
    with pytest.raises(ValueError, match="same-player current shin"):
        replace(observation, shin_clearance=replace(observation.shin_clearance, frame=19))
