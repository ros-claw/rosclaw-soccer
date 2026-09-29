"""Measured routing never uses a course label or a future success outcome."""

from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.receiving_measured_skill_router import (
    ReceivingMeasuredSkillRouter,
    ReceivingSkillKnot,
)
from rosclaw_soccer.rsi.receiving_middle_basis_expert import ReceivingMiddleBasisExpert
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _actor() -> ReceivingMeasuredSkillRouter:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    return ReceivingMeasuredSkillRouter(
        "red.finisher",
        schedule.contract_hash,
        ReceiveContactMailbox("red.finisher"),
        (0.0,) * 8,
        (0.0,) * 12,
        (0.0,) * 12,
        skill_knots=(ReceivingSkillKnot(0.138, -0.9, "high", (0.1,) * 12),),
    )


def _observation(lateral: float, closing: float) -> SimpleNamespace:
    qpos = [0.0] * 43
    qvel = [0.0] * 41
    qpos[2] = 0.7
    qpos[36:39] = [0.4, lateral, 0.1]
    qvel[35] = closing
    return SimpleNamespace(
        frame=15,
        qpos=tuple(qpos),
        qvel=tuple(qvel),
        last_own_foot_contact_time_sec=None,
    )


def test_router_selects_once_from_measured_initial_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ReceivingMiddleBasisExpert, "propose", lambda self, observation: (0.0,) * 29
    )
    actor = _actor()
    actor.propose(_observation(0.138, -0.9))
    assert actor.selected_expert == "high"
    assert actor.middle_weights == (0.1,) * 12
    actor.propose(_observation(0.17, -1.6))
    assert actor.selected_expert == "high"


def test_router_falls_back_outside_certified_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ReceivingMiddleBasisExpert, "propose", lambda self, observation: (0.0,) * 29
    )
    actor = _actor()
    actor.propose(_observation(0.17, -1.6))
    assert actor.selected_expert == "parent"
    assert actor.middle_weights == (0.0,) * 12
