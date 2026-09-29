"""Left/right A1 receiver symmetry uses only completed causal observations."""

from dataclasses import replace

import numpy as np

from rosclaw_soccer.rsi.receiving_mirrored_body_residual import (
    _MIRROR_ORDER,
    _MIRROR_SIGN,
    ReceivingMirroredBodyResidual,
)
from rosclaw_soccer.rsi.receiving_whole_body_residual import ReceivingWholeBodyResidual
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactMailbox,
    ReceiveContactSnapshot,
)
from rosclaw_soccer.training.receiving_feedback import ReceivingContactHistory
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from tests.training.test_receiving_feedback import observation


def test_completed_right_foot_mirrors_same_left_foot_actor():
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))
    coefficients = (0.5,) * 12
    left_mailbox = ReceiveContactMailbox("red.finisher")
    right_mailbox = ReceiveContactMailbox("red.finisher")
    left_mailbox._snapshot = ReceiveContactSnapshot(0.4, 0.38, "left_foot", 0.0, 0)
    right_mailbox._snapshot = ReceiveContactSnapshot(0.4, 0.38, "right_foot", 0.0, 0)
    left = ReceivingWholeBodyResidual(
        "red.finisher", schedule.contract_hash, left_mailbox, coefficients
    )
    right = ReceivingMirroredBodyResidual(
        "red.finisher", schedule.contract_hash, right_mailbox, coefficients
    )
    value = replace(
        observation(20),
        agent_id="red.finisher",
        action_substrate="A1_body29",
        previous_body_residual_rad=(0.0,) * 29,
        contact_history=ReceivingContactHistory(0.4, None),
    )
    expected = np.asarray(left.propose(value))[_MIRROR_ORDER] * _MIRROR_SIGN
    actual = right.propose(value)
    assert np.allclose(actual, expected, atol=1e-12)
    assert max(abs(x) for x in actual) <= 0.1
    assert right.contract_hash != left.contract_hash


def test_current_ball_side_selects_precontact_mirror_without_future_touch():
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))
    mailbox = ReceiveContactMailbox("red.finisher")
    mailbox._snapshot = ReceiveContactSnapshot(0.4, None, None, None, 0)
    coefficients = (0.6,) * 6 + (0.0,) * 6
    left = ReceivingMirroredBodyResidual(
        "red.finisher", schedule.contract_hash, mailbox, coefficients
    )
    right = ReceivingMirroredBodyResidual(
        "red.finisher", schedule.contract_hash, mailbox, coefficients
    )
    value = replace(
        observation(20),
        agent_id="red.finisher",
        action_substrate="A1_body29",
        previous_body_residual_rad=(0.0,) * 29,
        contact_history=ReceivingContactHistory(0.4, None),
    )
    left_q = list(value.qpos)
    right_q = list(value.qpos)
    left_q[37] = left_q[1] + 0.08
    right_q[37] = right_q[1] - 0.08
    left_action = np.asarray(left.propose(replace(value, qpos=tuple(left_q))))
    right_action = np.asarray(right.propose(replace(value, qpos=tuple(right_q))))
    assert np.allclose(right_action, left_action[_MIRROR_ORDER] * _MIRROR_SIGN, atol=1e-12)
