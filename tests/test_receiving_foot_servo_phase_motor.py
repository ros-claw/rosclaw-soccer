"""Bounded measured foot feedback stays inside the existing A2 motion proposal."""

from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.receiving_adaptive_phase_motor import ReceivingAdaptivePhaseMotor
from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_foot_servo_phase_motor import ReceivingFootServoPhaseMotor
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _actor(position: float, velocity: float = 0.0) -> ReceivingFootServoPhaseMotor:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    ref = ReceivingFootPhaseReference("high", 0.138, -0.9, 30, ((0.0,) * 48,) * 50)
    return ReceivingFootServoPhaseMotor(
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
        servo_position_gain=position,
        servo_velocity_horizon_sec=velocity,
    )


def _observation(foot: int = 1) -> SimpleNamespace:
    jacobian = (((1.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)),) * 2
    return SimpleNamespace(
        time_sec=0.48,
        last_own_contact_foot=foot,
        qpos=(0.0,) * 36 + (0.20, 0.0, 0.0) + (0.0,) * 4,
        qvel=(0.0,) * 35 + (0.0, 0.0, 0.0) + (0.0,) * 3,
        foot_kinematics=SimpleNamespace(
            foot_position_world_m=((0.0, 0.0, 0.0),) * 2,
            foot_linear_velocity_world_mps=((0.0, 0.0, 0.0),) * 2,
            foot_linear_jacobian_world=jacobian,
        ),
    )


def test_zero_servo_is_exact_parent_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ReceivingAdaptivePhaseMotor,
        "propose",
        lambda self, observation: (0.123,) * 29,
    )
    actor = _actor(0.0)
    actor.selected_expert = "high"
    actor.first_contact_time_sec = 0.4
    assert actor.propose(_observation()) == (0.123,) * 29
    assert actor.servo_active_frames == 0


def test_measured_servo_is_bounded_and_contact_foot_is_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ReceivingAdaptivePhaseMotor,
        "propose",
        lambda self, observation: (0.0,) * 29,
    )
    actor = _actor(0.5)
    actor.selected_expert = "high"
    actor.first_contact_time_sec = 0.4
    action = actor.propose(_observation(1))
    assert 0 < action[0] <= 0.10
    assert action[1:] == (0.0,) * 28
    assert actor.servo_foot_index == 0
    assert actor.propose(_observation(2)) == action
    assert actor.servo_active_frames == 2


def test_protected_and_missing_measurement_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ReceivingAdaptivePhaseMotor,
        "propose",
        lambda self, observation: (0.0,) * 29,
    )
    actor = _actor(0.3)
    actor.selected_expert = "high"
    actor.first_contact_time_sec = 0.4
    actor.protected_episode = True
    assert actor.propose(_observation()) == (0.0,) * 29
    actor.protected_episode = False
    obs = _observation()
    obs.foot_kinematics = None
    with pytest.raises(ValueError, match="same-frame measured foot"):
        actor.propose(obs)
    with pytest.raises(ValueError, match="bounded finite"):
        _actor(0.51)
