"""Skill recall is state-based, disjoint from retained parent, and episode-frozen."""

from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.receiving_composed_contact_router import ReceivingComposedContactRouter
from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_growing_skill_bank import ReceivingGrowingSkillBank
from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import (
    KinematicMotorWeights,
    ReceivingKinematicTemporalExpert,
)
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _actor(anchor: tuple[float, ...] = (0.0,) * 10) -> ReceivingGrowingSkillBank:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    ref = ReceivingFootPhaseReference("high", 0.138, -0.9, 30, ((0.0,) * 48,) * 50)
    return ReceivingGrowingSkillBank(
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
        neural_weights=KinematicMotorWeights(output_bias=(0.1,) * 12),
        protected_initial_features=((0.1,) * 10,) * 6,
        learned_weights=KinematicMotorWeights(output_bias=(0.4,) * 12),
        learned_initial_features=anchor,
    )


def test_new_skill_selected_without_weakening_old_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def base(
        self: ReceivingComposedContactRouter, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self.selected_expert = "high"
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingComposedContactRouter, "propose", base)
    monkeypatch.setattr(
        ReceivingKinematicTemporalExpert,
        "features",
        staticmethod(lambda observation: (observation.value,) * 10 + (0.0,) * 38),
    )
    learned = _actor()
    learned_action = learned.propose(SimpleNamespace(frame=19, value=0.0))[0]
    assert learned.learned_episode is True
    assert learned.protected_episode is False
    assert learned_action > 0
    assert learned.propose(SimpleNamespace(frame=25, value=0.1))[0] == learned_action

    baseline = _actor()
    baseline_action = baseline.propose(SimpleNamespace(frame=19, value=0.05))[0]
    assert baseline.learned_episode is False
    assert 0 < baseline_action < learned_action

    protected = _actor()
    assert protected.propose(SimpleNamespace(frame=19, value=0.1)) == (0.0,) * 29
    assert protected.protected_episode is True
    assert protected.learned_episode is False


def test_learned_region_cannot_overlap_original_protection() -> None:
    with pytest.raises(ValueError, match="disjoint bounded"):
        _actor((0.1,) * 10)
