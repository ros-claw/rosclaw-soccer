"""Composed local skills preserve single measured-state routing authority."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.receiving_composed_contact_router import ReceivingComposedContactRouter
from rosclaw_soccer.rsi.receiving_foot_phase_router import (
    ReceivingFootPhaseReference,
    ReceivingFootPhaseRouter,
)
from rosclaw_soccer.rsi.receiving_measured_skill_router import ReceivingSkillKnot
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _actor() -> ReceivingComposedContactRouter:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    high_ref = ReceivingFootPhaseReference("high", 0.138, -0.9, 30, ((0.0,) * 48,) * 50)
    return ReceivingComposedContactRouter(
        "red.finisher",
        schedule.contract_hash,
        ReceiveContactMailbox("red.finisher"),
        (0.0,) * 8,
        (0.0,) * 12,
        (0.0,) * 12,
        skill_knots=(ReceivingSkillKnot(0.138, -0.9, "high", (0.1,) * 12),),
        references=(high_ref,),
        corrected_experts=("high",),
        foot_gain=0.3,
        low_post_weights=(0.2,) * 12,
    )


def test_parent_and_center_paths_do_not_receive_low_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs = SimpleNamespace(time_sec=0.70, last_own_foot_contact_time_sec=0.60)

    def parent(self: ReceivingFootPhaseRouter, observation: SimpleNamespace) -> tuple[float, ...]:
        self.selected_expert = "parent"
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingFootPhaseRouter, "propose", parent)
    actor = _actor()
    assert actor.propose(obs) == (0.0,) * 29

    def center(self: ReceivingFootPhaseRouter, observation: SimpleNamespace) -> tuple[float, ...]:
        self.selected_expert = "center"
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingFootPhaseRouter, "propose", center)
    actor = _actor()
    assert actor.propose(obs) == (0.0,) * 29
    assert actor.low_active_frames == 0


def test_low_support_is_contact_triggered_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    def low(self: ReceivingFootPhaseRouter, observation: SimpleNamespace) -> tuple[float, ...]:
        self.selected_expert = "low"
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingFootPhaseRouter, "propose", low)
    actor = _actor()
    assert (
        actor.propose(SimpleNamespace(time_sec=0.54, last_own_foot_contact_time_sec=None))
        == (0.0,) * 29
    )
    target = actor.propose(SimpleNamespace(time_sec=0.70, last_own_foot_contact_time_sec=0.60))
    assert target[:12] == pytest.approx((0.05,) * 12)
    assert actor.low_active_frames == 1


def test_composition_requires_high_only_correction() -> None:
    actor = _actor()
    with pytest.raises(ValueError, match="high-only"):
        replace(actor, corrected_experts=("low", "high"))
