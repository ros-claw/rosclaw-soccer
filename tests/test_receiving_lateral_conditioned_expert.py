"""Measured-lateral receiving stays on one bounded SIM_ONLY A2 authority."""

import pytest

from rosclaw_soccer.rsi.receiving_lateral_conditioned_expert import (
    ReceivingLateralConditionedExpert,
)
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def test_lateral_actor_contract_and_invalid_slope() -> None:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox("red.finisher")
    args = (
        "red.finisher",
        schedule.contract_hash,
        mailbox,
        (0.0,) * 8,
        (0.0,) * 12,
        (0.0,) * 12,
    )
    actor = ReceivingLateralConditionedExpert(*args)
    assert actor.action_substrate == "A2_body29_precontact"
    assert actor.activation_ceiling == "SIM_ONLY"
    assert actor.measured_lateral_m is None
    with pytest.raises(ValueError):
        ReceivingLateralConditionedExpert(
            *args,
            left_lateral_slope=(float("nan"),) + (0.0,) * 11,
        )
