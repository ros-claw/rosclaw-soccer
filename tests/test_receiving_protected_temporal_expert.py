"""A predeclared measured retention state selects the exact parent path."""

from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.receiving_lateral_piecewise_expert import ReceivingLateralPiecewiseExpert
from rosclaw_soccer.rsi.receiving_protected_temporal_expert import (
    ReceivingProtectedTemporalExpert,
)
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import (
    ReceivingTemporalMotorExpert,
    TemporalMotorWeights,
)
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _observation(y: float) -> SimpleNamespace:
    qpos = [0.0] * 43
    qvel = [0.0] * 41
    qpos[2] = 0.7
    qpos[36:39] = [0.4, y, 0.1]
    qvel[35] = -1.25
    return SimpleNamespace(
        qpos=tuple(qpos),
        qvel=tuple(qvel),
        frame=20,
        last_own_foot_contact_time_sec=None,
    )


def _actor(protected: tuple[tuple[float, ...], ...]) -> ReceivingProtectedTemporalExpert:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    return ReceivingProtectedTemporalExpert(
        "red.finisher",
        schedule.contract_hash,
        ReceiveContactMailbox("red.finisher"),
        (0.0,) * 8,
        (0.0,) * 12,
        (0.0,) * 12,
        policy=TemporalMotorWeights(output_bias=(0.2,) + (0.0,) * 11),
        protected_initial_features=protected,
    )


def test_protected_anchor_uses_parent_without_temporal_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def base(
        self: ReceivingLateralPiecewiseExpert, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self._selected_side = 0
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingLateralPiecewiseExpert, "propose", base)
    first = ReceivingTemporalMotorExpert.features(_observation(0.14))
    second = ReceivingTemporalMotorExpert.features(_observation(0.16))
    actor = _actor((first, second))
    assert actor.propose(_observation(0.14)) == (0.0,) * 29
    assert actor.protected_episode is True
    assert actor.observed_frames == []


def test_non_anchor_keeps_temporal_learning_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def base(
        self: ReceivingLateralPiecewiseExpert, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self._selected_side = 0
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingLateralPiecewiseExpert, "propose", base)
    first = ReceivingTemporalMotorExpert.features(_observation(0.14))
    second = ReceivingTemporalMotorExpert.features(_observation(0.16))
    actor = _actor((first, second))
    output = actor.propose(_observation(0.15))
    assert actor.protected_episode is False
    assert output[0] > 0
    assert actor.observed_frames == [20]


def test_invalid_parent_memory_fails_closed() -> None:
    with pytest.raises(ValueError, match="two finite bounded"):
        _actor(((0.0,) * 10,))
