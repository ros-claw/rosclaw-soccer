"""Zero-action whole-body receiving tap keeps contact and motor authority separate."""

from dataclasses import replace

import pytest

from rosclaw_soccer.rsi.receiving_whole_body_contact_tap import ReceivingWholeBodyContactTap
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.skills.team.motor_option import TeamMotorPhysicsObservation
from rosclaw_soccer.training.receiving_feedback import ReceivingContactHistory
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from tests.training.test_receiving_feedback import observation


def _physics(time_sec: float) -> TeamMotorPhysicsObservation:
    return TeamMotorPhysicsObservation(
        time_sec,
        (0.0,) * 43,
        (0.0,) * 41,
        True,
        0.0,
        0.0,
        observer_agent_id="red.finisher",
        contacts_complete=True,
    )


def _observation(frame: int):
    value = observation(frame)
    return replace(
        value,
        agent_id="red.finisher",
        action_substrate="A1_body29",
        previous_body_residual_rad=(0.0,) * 29,
        contact_history=ReceivingContactHistory(frame * 0.02, None),
    )


def test_full_body_tap_is_zero_only_identity_bound_and_clocked():
    mailbox = ReceiveContactMailbox("red.finisher")
    schedule = ReceivingOracleSchedule("red.finisher", "A1_body29", 20, 10, ((0.0,) * 29,))
    tap = ReceivingWholeBodyContactTap("red.finisher", schedule.contract_hash, mailbox)
    assert not hasattr(tap, "observe_physics") and not hasattr(tap, "motor_target")
    for step in range(1, 201):
        mailbox.consume(_physics(step * 0.002))
    assert tap.propose(_observation(20)) == (0.0,) * 29
    arrays = tap.arrays()
    assert arrays["qpos"].shape == (1, 43)
    assert arrays["qvel"].shape == (1, 41)
    assert arrays["foundation_target_rad"].shape == (1, 29)
    assert arrays["previous_body_residual_rad"].shape == (1, 29)
    with pytest.raises(ValueError, match="consecutive completed"):
        tap.propose(_observation(20))
    with pytest.raises(ValueError):
        ReceivingWholeBodyContactTap("blue.finisher", schedule.contract_hash, mailbox)
