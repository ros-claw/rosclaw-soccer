"""Measured hard retention bypasses a learned residual for the whole episode."""

from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.receiving_composed_contact_router import ReceivingComposedContactRouter
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


def _actor(protected: tuple[tuple[float, ...], ...]) -> ReceivingProtectedComposedNeural:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    ref = ReceivingFootPhaseReference("high", 0.138, -0.9, 30, ((0.0,) * 48,) * 50)
    return ReceivingProtectedComposedNeural(
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
        neural_weights=KinematicMotorWeights(output_bias=(0.2,) * 12),
        protected_initial_features=protected,
    )


def test_matching_first_state_bypasses_nonzero_neural_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def base(
        self: ReceivingComposedContactRouter, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self.selected_expert = "high"
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingComposedContactRouter, "propose", base)
    monkeypatch.setattr(
        ReceivingKinematicTemporalExpert, "features", staticmethod(lambda observation: (0.0,) * 48)
    )
    actor = _actor(((0.0,) * 10,) * 6)
    assert actor.propose(SimpleNamespace(frame=19)) == (0.0,) * 29
    assert actor.protected_episode is True
    assert actor.propose(SimpleNamespace(frame=25)) == (0.0,) * 29
    assert actor.observed_frames == []


def test_unmatched_state_can_learn_and_protection_is_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def base(
        self: ReceivingComposedContactRouter, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self.selected_expert = "high"
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingComposedContactRouter, "propose", base)
    monkeypatch.setattr(
        ReceivingKinematicTemporalExpert, "features", staticmethod(lambda observation: (0.0,) * 48)
    )
    actor = _actor(((0.1,) * 10,) * 6)
    assert actor.propose(SimpleNamespace(frame=19))[0] > 0
    assert actor.protected_episode is False
    assert actor.propose(SimpleNamespace(frame=25))[0] > 0


def test_requires_six_finite_retention_states() -> None:
    with pytest.raises(ValueError, match="six finite measured"):
        _actor(((0.0,) * 10,))
