"""Measured-state neural residual has bounded output and frozen episode context."""

from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi.receiving_contextual_motor_expert import (
    ContextualMotorWeights,
    ReceivingContextualMotorExpert,
)
from rosclaw_soccer.rsi.receiving_lateral_piecewise_expert import ReceivingLateralPiecewiseExpert
from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule


def _actor(**kwargs: object) -> ReceivingContextualMotorExpert:
    schedule = ReceivingOracleSchedule(
        "red.finisher", "A2_body29_precontact", 15, 10, ((0.0,) * 29,)
    )
    return ReceivingContextualMotorExpert(
        "red.finisher",
        schedule.contract_hash,
        ReceiveContactMailbox("red.finisher"),
        (0.0,) * 8,
        (0.0,) * 12,
        (0.0,) * 12,
        policy=ContextualMotorWeights(output_bias=(0.2,) + (0.0,) * 11),
        **kwargs,
    )


def _observation(lateral: float, speed: float, frame: int) -> SimpleNamespace:
    qpos = [0.0] * 43
    qvel = [0.0] * 41
    qpos[37] = lateral
    qvel[35] = -speed
    return SimpleNamespace(qpos=tuple(qpos), qvel=tuple(qvel), frame=frame)


def test_contextual_policy_rejects_nonfinite_and_wrong_shape() -> None:
    with pytest.raises(ValueError, match="finite bounded contextual"):
        ContextualMotorWeights(output_bias=(float("nan"),) + (0.0,) * 11)
    with pytest.raises(ValueError, match="finite bounded contextual"):
        ContextualMotorWeights(input_matrix=(0.0,) * 31)
    with pytest.raises(ValueError, match="bounded SIM_ONLY contextual"):
        _actor(exploration_noise=(float("inf"),) + (0.0,) * 11)


def test_context_is_measured_once_and_action_has_fixed_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def base(
        self: ReceivingLateralPiecewiseExpert, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self._selected_side = 0
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingLateralPiecewiseExpert, "propose", base)
    actor = _actor()
    start = actor.propose(_observation(0.14, 1.25, 15))
    peak = actor.propose(_observation(0.20, 3.0, 20))
    end = actor.propose(_observation(0.20, 3.0, 41))
    assert actor.measured_context == pytest.approx((0.0, 0.0))
    assert actor.activation_ceiling == "SIM_ONLY"
    assert actor.selected_raw_action == pytest.approx((0.2,) + (0.0,) * 11)
    assert start == pytest.approx((0.0,) * 29)
    assert peak[0] == pytest.approx(0.25 * 0.197375320224904)
    assert end == pytest.approx((0.0,) * 29)


def test_nonfinite_measured_velocity_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def base(
        self: ReceivingLateralPiecewiseExpert, observation: SimpleNamespace
    ) -> tuple[float, ...]:
        self._selected_side = 0
        return (0.0,) * 29

    monkeypatch.setattr(ReceivingLateralPiecewiseExpert, "propose", base)
    with pytest.raises(ValueError, match="finite current"):
        _actor().propose(_observation(0.14, float("nan"), 15))
