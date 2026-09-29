"""Measured phase activation is episode-frozen and leaves other states unchanged."""

from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_gated_phase_contact_motor import (
    ReceivingGatedPhaseContactMotor,
)
from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import (
    ReceivingKinematicTemporalExpert,
)
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.rsi.receiving_phase_contact_motor import ReceivingPhaseContactMotor
from rosclaw_soccer.rsi.receiving_protected_composed_neural import (
    ReceivingProtectedComposedNeural,
)
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _actor() -> ReceivingGatedPhaseContactMotor:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    ref = ReceivingFootPhaseReference("high", 0.138, -0.9, 30, ((0.0,) * 48,) * 50)
    return ReceivingGatedPhaseContactMotor(
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
        post_contact_logits=(1.0,) * 12,
        activation_states=((0.0,) * 10, (0.04,) * 10),
    )


def test_activation_is_measured_and_episode_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    def base(
        self: ReceivingProtectedComposedNeural, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self.selected_expert = "high"
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingProtectedComposedNeural, "propose", base)
    monkeypatch.setattr(
        ReceivingKinematicTemporalExpert,
        "features",
        staticmethod(lambda observation: (observation.value,) * 10 + (0.0,) * 38),
    )
    selected = _actor()
    assert (
        selected.propose(
            SimpleNamespace(frame=19, value=0.0, time_sec=0.38, last_own_foot_contact_time_sec=None)
        )
        == (0.0,) * 29
    )
    assert selected.activation_enabled
    assert (
        selected.propose(
            SimpleNamespace(frame=24, value=0.2, time_sec=0.48, last_own_foot_contact_time_sec=0.40)
        )[0]
        > 0
    )
    rejected = _actor()
    assert (
        rejected.propose(
            SimpleNamespace(frame=19, value=0.2, time_sec=0.38, last_own_foot_contact_time_sec=None)
        )
        == (0.0,) * 29
    )
    assert not rejected.activation_enabled
    assert (
        rejected.propose(
            SimpleNamespace(frame=24, value=0.0, time_sec=0.48, last_own_foot_contact_time_sec=0.40)
        )
        == (0.0,) * 29
    )
    assert rejected.post_contact_logits == (0.0,) * 12


def test_gate_requires_two_finite_states() -> None:
    actor = _actor()
    actor.activation_states = ((0.0,) * 10,)
    with pytest.raises(ValueError, match="two finite"):
        actor.__post_init__()


def test_parent_zero_action_still_exact_with_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ReceivingProtectedComposedNeural,
        "propose",
        lambda self, observation: (0.0,) * 29,
    )
    monkeypatch.setattr(
        ReceivingKinematicTemporalExpert,
        "features",
        staticmethod(lambda observation: (0.0,) * 48),
    )
    actor = _actor()
    actor.post_contact_logits = (0.0,) * 12
    assert isinstance(actor, ReceivingPhaseContactMotor)
    assert (
        actor.propose(SimpleNamespace(frame=19, time_sec=0.38, last_own_foot_contact_time_sec=None))
        == (0.0,) * 29
    )
