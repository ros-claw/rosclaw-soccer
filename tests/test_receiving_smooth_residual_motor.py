"""Continuous residual gain retains the exact parent outside the measured ramp."""

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
from test_receiving_lateral_velocity_gated_precontact_motor import _actor as _lateral_actor

from rosclaw_soccer.rsi.receiving_foot_servo_phase_motor import ReceivingFootServoPhaseMotor
from rosclaw_soccer.rsi.receiving_smooth_residual_motor import ReceivingSmoothResidualMotor
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import ReceivingTemporalMotorExpert


def _actor() -> ReceivingSmoothResidualMotor:
    base = _lateral_actor()
    values = {field.name: getattr(base, field.name) for field in fields(base) if field.init}
    values["maximum_relative_lateral_feature"] = 0.7696218896207085
    values["activation_states"] = ((0.0, 0.766, 0.0, -0.45) + (0.0,) * 6,) * 3
    weights = base.candidate_neural_weights
    values["refined_neural_weights"] = replace(
        weights,
        output_bias=tuple(value + 0.02 for value in weights.output_bias),
    )
    return ReceivingSmoothResidualMotor(**values)


def test_continuous_gain_is_bounded_and_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ReceivingFootServoPhaseMotor,
        "propose",
        lambda self, observation: (0.0,) * 29,
    )
    monkeypatch.setattr(
        ReceivingTemporalMotorExpert,
        "features",
        staticmethod(
            lambda observation: (
                0.0,
                observation.lateral_feature,
                0.0,
                observation.speed_feature,
                *(0.0,) * 6,
            )
        ),
    )
    zero = _actor()
    zero.propose(SimpleNamespace(frame=19, lateral_feature=0.764, speed_feature=-0.45))
    assert zero.activation_enabled and zero.residual_gain == 0
    assert zero.neural_weights == zero.candidate_neural_weights

    middle = _actor()
    middle.propose(SimpleNamespace(frame=19, lateral_feature=0.766, speed_feature=-0.45))
    assert middle.residual_gain == pytest.approx(0.5)
    assert middle.neural_weights.output_bias == pytest.approx(
        tuple(value + 0.01 for value in middle.candidate_neural_weights.output_bias)
    )
    middle.propose(SimpleNamespace(frame=20, lateral_feature=0.75, speed_feature=-0.48))
    assert middle.residual_gain == pytest.approx(0.5)

    full = _actor()
    full.propose(SimpleNamespace(frame=19, lateral_feature=0.768, speed_feature=-0.45))
    assert full.residual_gain == 1
    assert full.neural_weights == full.refined_neural_weights


@pytest.mark.parametrize(
    "field,value",
    [
        ("gain_lateral_start", 0.768),
        ("gain_lateral_full", 0.776),
        ("gain_lateral_start", float("nan")),
        ("gain_lateral_full", True),
    ],
)
def test_invalid_gain_ramps_fail_closed(field: str, value: float) -> None:
    actor = _actor()
    setattr(actor, field, value)
    with pytest.raises(ValueError, match="bounded compatible measured-state"):
        actor.__post_init__()
