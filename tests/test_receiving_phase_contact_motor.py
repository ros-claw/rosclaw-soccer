"""Measured post-contact motor phase stays bounded, smooth, and zero-equivalent."""

from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.rsi.receiving_phase_contact_motor import ReceivingPhaseContactMotor
from rosclaw_soccer.rsi.receiving_protected_composed_neural import (
    ReceivingProtectedComposedNeural,
)
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _actor(logits: tuple[float, ...]) -> ReceivingPhaseContactMotor:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    ref = ReceivingFootPhaseReference("high", 0.138, -0.9, 30, ((0.0,) * 48,) * 50)
    return ReceivingPhaseContactMotor(
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
        post_contact_logits=logits,
    )


def test_post_phase_requires_measured_touch_and_fades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def base(
        self: ReceivingProtectedComposedNeural, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self.selected_expert = "high"
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingProtectedComposedNeural, "propose", base)
    actor = _actor((1.0,) * 12)
    assert (
        actor.propose(SimpleNamespace(time_sec=0.38, last_own_foot_contact_time_sec=None))
        == (0.0,) * 29
    )
    early = actor.propose(SimpleNamespace(time_sec=0.40, last_own_foot_contact_time_sec=0.38))[0]
    peak = actor.propose(SimpleNamespace(time_sec=0.48, last_own_foot_contact_time_sec=0.46))[0]
    late = actor.propose(SimpleNamespace(time_sec=0.64, last_own_foot_contact_time_sec=0.62))[0]
    assert 0 < early < peak < 0.15
    assert 0 < late < peak
    assert (
        actor.propose(SimpleNamespace(time_sec=0.72, last_own_foot_contact_time_sec=0.70))
        == (0.0,) * 29
    )
    assert actor.first_contact_time_sec == 0.38


def test_zero_logits_and_protected_episode_are_exact_noops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ReceivingProtectedComposedNeural,
        "propose",
        lambda self, observation: (0.0,) * 29,
    )
    zero = _actor((0.0,) * 12)
    assert (
        zero.propose(SimpleNamespace(time_sec=0.50, last_own_foot_contact_time_sec=0.40))
        == (0.0,) * 29
    )
    assert zero.first_contact_time_sec is None
    guarded = _actor((1.0,) * 12)
    guarded.protected_episode = True
    assert (
        guarded.propose(SimpleNamespace(time_sec=0.50, last_own_foot_contact_time_sec=0.40))
        == (0.0,) * 29
    )
