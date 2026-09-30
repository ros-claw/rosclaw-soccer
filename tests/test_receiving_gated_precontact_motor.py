"""First-frame measured gate isolates a learned precontact motor skill."""

from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_foot_servo_phase_motor import ReceivingFootServoPhaseMotor
from rosclaw_soccer.rsi.receiving_gated_precontact_motor import ReceivingGatedPrecontactMotor
from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import KinematicMotorWeights
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import ReceivingTemporalMotorExpert
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _actor() -> ReceivingGatedPrecontactMotor:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    ref = ReceivingFootPhaseReference("high", 0.138, -0.9, 30, ((0.0,) * 48,) * 50)
    return ReceivingGatedPrecontactMotor(
        "red.finisher",
        schedule.contract_hash,
        ReceiveContactMailbox("red.finisher"),
        (0.0,) * 8,
        (0.0,) * 12,
        (0.0,) * 12,
        skill_knots=(ReceivingSkillKnot(0.138, -0.9, "high", (0.1,) * 12),),
        references=(ref,),
        corrected_experts=("high",),
        foot_gain=0.3,
        low_post_weights=(0.0,) * 12,
        protected_initial_features=((0.1,) * 10,) * 6,
        candidate_neural_weights=KinematicMotorWeights(output_bias=(0.2,) * 12),
        activation_states=((0.0,) * 10,) * 3,
    )


def test_first_state_gate_is_frozen_and_parent_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ReceivingFootServoPhaseMotor,
        "propose",
        lambda self, observation: (0.0,) * 29,
    )
    monkeypatch.setattr(
        ReceivingTemporalMotorExpert,
        "features",
        staticmethod(lambda observation: (observation.marker,) + (0.0,) * 9),
    )
    actor = _actor()
    original = actor.neural_weights
    actor.propose(SimpleNamespace(frame=18, marker=0.0))
    assert not actor.activation_decided
    actor.propose(SimpleNamespace(frame=19, marker=0.005))
    assert actor.activation_enabled
    assert actor.neural_weights == actor.candidate_neural_weights
    actor.propose(SimpleNamespace(frame=20, marker=1.0))
    assert actor.activation_enabled

    other = _actor()
    other.propose(SimpleNamespace(frame=19, marker=0.018))
    assert other.activation_decided and not other.activation_enabled
    assert other.neural_weights == original


def test_gate_requires_three_finite_states() -> None:
    actor = _actor()
    actor.activation_states = ((float("nan"),) + (0.0,) * 9,) * 3
    with pytest.raises(ValueError, match="three finite"):
        actor.__post_init__()
