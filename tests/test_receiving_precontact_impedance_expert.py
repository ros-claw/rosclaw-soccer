"""Bounded first-touch impedance shares one guarded A2 authority."""

import pytest

from rosclaw_soccer.rsi.receiving_precontact_impedance_expert import (
    ReceivingPrecontactImpedanceExpert,
)
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def test_precontact_impedance_bound_and_same_player_contract() -> None:
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
    actor = ReceivingPrecontactImpedanceExpert(*args)
    assert actor.action_substrate == "A2_body29_precontact"
    assert actor.activation_ceiling == "SIM_ONLY"
    with pytest.raises(ValueError):
        ReceivingPrecontactImpedanceExpert(*args, left_pre_gain=float("nan"))
    with pytest.raises(ValueError):
        ReceivingPrecontactImpedanceExpert(*args, left_pre_gain=1.01)
