"""Multiple local learned policies coexist without weakening old skills."""

from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.receiving_composed_contact_router import ReceivingComposedContactRouter
from rosclaw_soccer.rsi.receiving_foot_phase_router import ReceivingFootPhaseReference
from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import (
    KinematicMotorWeights,
    ReceivingKinematicTemporalExpert,
)
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.rsi.receiving_multi_skill_bank import (
    ReceivingMeasuredSkill,
    ReceivingMultiSkillBank,
)
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _actor(second_value: float = 0.04) -> ReceivingMultiSkillBank:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    ref = ReceivingFootPhaseReference("high", 0.138, -0.9, 30, ((0.0,) * 48,) * 50)
    return ReceivingMultiSkillBank(
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
        learned_skills=(
            ReceivingMeasuredSkill((0.0,) * 10, KinematicMotorWeights(output_bias=(0.2,) * 12)),
            ReceivingMeasuredSkill(
                (second_value,) * 10, KinematicMotorWeights(output_bias=(0.4,) * 12)
            ),
        ),
    )


def test_two_measured_skills_and_old_guard(monkeypatch: pytest.MonkeyPatch) -> None:
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
    first = _actor()
    first_action = first.propose(SimpleNamespace(frame=19, value=0.0))[0]
    assert first.selected_skill == 0
    assert first.propose(SimpleNamespace(frame=25, value=0.04))[0] == first_action
    second = _actor()
    second_action = second.propose(SimpleNamespace(frame=19, value=0.04))[0]
    assert second.selected_skill == 1
    assert second_action > first_action > 0
    unchanged = _actor()
    default_action = unchanged.propose(SimpleNamespace(frame=19, value=0.07))[0]
    assert unchanged.selected_skill is None
    assert 0 < default_action < first_action
    protected = _actor()
    assert protected.propose(SimpleNamespace(frame=19, value=0.1)) == (0.0,) * 29
    assert protected.protected_episode is True


def test_skill_regions_must_not_overlap() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        _actor(0.002)
    with pytest.raises(ValueError, match="disjoint"):
        _actor(0.1)
