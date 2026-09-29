"""Measured-touch support learning cannot alter the protected parent path."""

from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.receiving_measured_skill_router import (
    ReceivingMeasuredSkillRouter,
    ReceivingSkillKnot,
)
from rosclaw_soccer.rsi.receiving_post_contact_residual import ReceivingPostContactResidual
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _actor(weights: tuple[float, ...]) -> ReceivingPostContactResidual:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    return ReceivingPostContactResidual(
        "red.finisher",
        schedule.contract_hash,
        ReceiveContactMailbox("red.finisher"),
        (0.0,) * 8,
        (0.0,) * 12,
        (0.0,) * 12,
        skill_knots=(ReceivingSkillKnot(0.138, -0.9, "high", (0.1,) * 12),),
        post_weights=weights,
    )


def test_post_contact_zero_action_and_parent_fallback_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def high(self: ReceivingMeasuredSkillRouter, observation: SimpleNamespace) -> tuple[float, ...]:
        self.selected_expert = "high"
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingMeasuredSkillRouter, "propose", high)
    obs = SimpleNamespace(time_sec=0.70, last_own_foot_contact_time_sec=0.60)
    zero = _actor((0.0,) * 12)
    assert zero.propose(obs) == (0.0,) * 29
    assert zero.active_frames == 0

    def parent(
        self: ReceivingMeasuredSkillRouter, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self.selected_expert = "parent"
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingMeasuredSkillRouter, "propose", parent)
    protected = _actor((0.2,) * 12)
    assert protected.propose(obs) == (0.0,) * 29
    assert protected.active_frames == 0


def test_post_contact_uses_measured_touch_and_bounded_joint_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def high(self: ReceivingMeasuredSkillRouter, observation: SimpleNamespace) -> tuple[float, ...]:
        self.selected_expert = "high"
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingMeasuredSkillRouter, "propose", high)
    actor = _actor((0.2,) * 12)
    before = SimpleNamespace(time_sec=0.54, last_own_foot_contact_time_sec=None)
    assert actor.propose(before) == (0.0,) * 29
    after = SimpleNamespace(time_sec=0.70, last_own_foot_contact_time_sec=0.60)
    assert actor.propose(after)[:12] == pytest.approx((0.05,) * 12)
    assert actor.active_frames == 1
    assert actor.peak_residual_rad == pytest.approx(0.05)


def test_post_contact_rejects_nonfinite_weights() -> None:
    with pytest.raises(ValueError, match="finite bounded"):
        _actor((float("nan"),) + (0.0,) * 11)
