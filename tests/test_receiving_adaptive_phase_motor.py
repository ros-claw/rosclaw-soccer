"""Online post-contact actor observes the body without bypassing old protection."""

from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.receiving_adaptive_phase_motor import ReceivingAdaptivePhaseMotor
from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import (
    KinematicMotorWeights,
    ReceivingKinematicTemporalExpert,
)
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.rsi.receiving_protected_composed_neural import (
    ReceivingProtectedComposedNeural,
)
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _actor(bias: float) -> ReceivingAdaptivePhaseMotor:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    ref = ReceivingFootPhaseReference("high", 0.138, -0.9, 30, ((0.0,) * 48,) * 50)
    return ReceivingAdaptivePhaseMotor(
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
        post_policy=KinematicMotorWeights(output_bias=(bias,) * 12),
    )


def test_post_actor_is_measured_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    def base(
        self: ReceivingProtectedComposedNeural, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self.selected_expert = "high"
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingProtectedComposedNeural, "propose", base)
    monkeypatch.setattr(
        ReceivingKinematicTemporalExpert,
        "features",
        staticmethod(lambda observation: (0.0,) * 48),
    )
    actor = _actor(1.0)
    assert (
        actor.propose(SimpleNamespace(frame=19, time_sec=0.38, last_own_foot_contact_time_sec=None))
        == (0.0,) * 29
    )
    action = actor.propose(
        SimpleNamespace(frame=24, time_sec=0.48, last_own_foot_contact_time_sec=0.4)
    )
    assert 0 < action[0] < 0.15
    assert actor.post_observed_frames == [24]
    assert actor.post_observed_features == [(0.0,) * 48]
    assert actor.post_sampled_logits == [(1.0,) * 12]
    assert actor.post_peak_residual_rad == action[0]
    late_actor = _actor(1.0)
    assert (
        late_actor.propose(
            SimpleNamespace(frame=55, time_sec=1.10, last_own_foot_contact_time_sec=0.82)
        )[0]
        > 0
    )
    assert late_actor.post_observed_frames == [55]


def test_zero_and_protected_actions_are_exact_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    def base(
        self: ReceivingProtectedComposedNeural, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self.selected_expert = "high"
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingProtectedComposedNeural, "propose", base)
    monkeypatch.setattr(
        ReceivingKinematicTemporalExpert,
        "features",
        staticmethod(lambda observation: (0.0,) * 48),
    )
    zero = _actor(0.0)
    assert (
        zero.propose(SimpleNamespace(frame=24, time_sec=0.48, last_own_foot_contact_time_sec=0.4))
        == (0.0,) * 29
    )
    protected = _actor(1.0)
    protected.protected_episode = True
    assert (
        protected.propose(
            SimpleNamespace(frame=24, time_sec=0.48, last_own_foot_contact_time_sec=0.4)
        )
        == (0.0,) * 29
    )
    assert protected.post_observed_frames == []
