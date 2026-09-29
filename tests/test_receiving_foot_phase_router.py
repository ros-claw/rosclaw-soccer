"""Foot-phase corrections remain measured, bounded, and parent-retaining."""

from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.receiving_foot_phase_router import (
    ReceivingFootPhaseReference,
    ReceivingFootPhaseRouter,
)
from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import ReceivingKinematicTemporalExpert
from rosclaw_soccer.rsi.receiving_measured_skill_router import (
    ReceivingMeasuredSkillRouter,
    ReceivingSkillKnot,
)
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _reference() -> ReceivingFootPhaseReference:
    row = [0.0] * 48
    row[10] = 0.04
    return ReceivingFootPhaseReference(
        "high", 0.138, -0.9, 30, tuple(tuple(row) for _ in range(50))
    )


def _actor(
    gain: float, post_multiplier: float = 1.0, shin_guard_m: float = 0.0
) -> ReceivingFootPhaseRouter:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    return ReceivingFootPhaseRouter(
        "red.finisher",
        schedule.contract_hash,
        ReceiveContactMailbox("red.finisher"),
        (0.0,) * 8,
        (0.0,) * 12,
        (0.0,) * 12,
        skill_knots=(ReceivingSkillKnot(0.138, -0.9, "high", (0.1,) * 12),),
        references=(_reference(),),
        foot_gain=gain,
        post_multiplier=post_multiplier,
        shin_guard_m=shin_guard_m,
    )


def _observation() -> SimpleNamespace:
    jacobian = ((1.0, 0.0, 0.0, 0.0, 0.0, 0.0),) + ((0.0,) * 6,) * 2
    return SimpleNamespace(
        frame=25,
        last_own_foot_contact_time_sec=None,
        foot_kinematics=SimpleNamespace(
            foot_linear_velocity_world_mps=((0.0,) * 3,) * 2,
            foot_linear_jacobian_world=(jacobian, jacobian),
        ),
        shin_clearance=SimpleNamespace(clearance_m=(0.03, 0.30)),
    )


def test_foot_phase_zero_gain_and_parent_fallback_keep_exact_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def base(self: ReceivingMeasuredSkillRouter, observation: SimpleNamespace) -> tuple[float, ...]:
        self.selected_expert = "high"
        self._selected_side = 0
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingMeasuredSkillRouter, "propose", base)
    monkeypatch.setattr(
        ReceivingKinematicTemporalExpert, "features", staticmethod(lambda observation: (0.0,) * 48)
    )
    actor = _actor(0.0)
    assert actor.propose(_observation()) == (0.0,) * 29
    assert actor.nonzero_frames == 0

    def parent(
        self: ReceivingMeasuredSkillRouter, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self.selected_expert = "parent"
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingMeasuredSkillRouter, "propose", parent)
    actor = _actor(1.0)
    assert actor.propose(_observation()) == (0.0,) * 29
    assert actor.selected_reference is None


def test_foot_phase_measured_jacobian_and_bounded_target(monkeypatch: pytest.MonkeyPatch) -> None:
    def base(self: ReceivingMeasuredSkillRouter, observation: SimpleNamespace) -> tuple[float, ...]:
        self.selected_expert = "high"
        self._selected_side = 0
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingMeasuredSkillRouter, "propose", base)
    monkeypatch.setattr(
        ReceivingKinematicTemporalExpert, "features", staticmethod(lambda observation: (0.0,) * 48)
    )
    actor = _actor(1.0)
    target = actor.propose(_observation())
    assert 0 < target[0] <= 0.04
    assert target[1:] == (0.0,) * 28
    assert actor.nonzero_frames == 1
    assert actor.peak_correction_rad == target[0]


def test_foot_phase_rejects_corrupt_reference() -> None:
    with pytest.raises(ValueError, match="verified bounded"):
        ReceivingFootPhaseReference("high", 0.138, -0.9, 30, ((float("nan"),) * 48,) * 50)


def test_post_contact_phase_can_be_disabled_without_disabling_precontact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def base(self: ReceivingMeasuredSkillRouter, observation: SimpleNamespace) -> tuple[float, ...]:
        self.selected_expert = "high"
        self._selected_side = 0
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingMeasuredSkillRouter, "propose", base)
    monkeypatch.setattr(
        ReceivingKinematicTemporalExpert, "features", staticmethod(lambda observation: (0.0,) * 48)
    )
    actor = _actor(1.0, 0.0)
    first = _observation()
    assert actor.propose(first)[0] > 0
    after = _observation()
    after.frame = 31
    after.last_own_foot_contact_time_sec = 0.6
    assert actor.propose(after) == (0.0,) * 29


def test_shin_guard_suppresses_only_post_contact_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def base(self: ReceivingMeasuredSkillRouter, observation: SimpleNamespace) -> tuple[float, ...]:
        self.selected_expert = "high"
        self._selected_side = 0
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingMeasuredSkillRouter, "propose", base)
    monkeypatch.setattr(
        ReceivingKinematicTemporalExpert, "features", staticmethod(lambda observation: (0.0,) * 48)
    )
    actor = _actor(1.0, shin_guard_m=0.04)
    assert actor.propose(_observation())[0] > 0
    after = _observation()
    after.frame = 31
    after.last_own_foot_contact_time_sec = 0.6
    assert actor.propose(after) == (0.0,) * 29


def test_zero_shin_guard_is_exact_noop_even_for_negative_clearance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def base(self: ReceivingMeasuredSkillRouter, observation: SimpleNamespace) -> tuple[float, ...]:
        self.selected_expert = "high"
        self._selected_side = 0
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingMeasuredSkillRouter, "propose", base)
    monkeypatch.setattr(
        ReceivingKinematicTemporalExpert, "features", staticmethod(lambda observation: (0.0,) * 48)
    )
    actor = _actor(1.0, shin_guard_m=0.0)
    after = _observation()
    after.last_own_foot_contact_time_sec = 0.48
    after.shin_clearance = SimpleNamespace(clearance_m=(-0.01, 0.3))
    assert actor.propose(after)[0] > 0
