"""A measured contact band may refine a skill without replacing old regions."""

from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
from test_receiving_lateral_velocity_gated_precontact_motor import _actor as _lateral_actor

from rosclaw_soccer.rsi.receiving_band_refined_precontact_motor import (
    ReceivingBandRefinedPrecontactMotor,
)
from rosclaw_soccer.rsi.receiving_foot_servo_phase_motor import ReceivingFootServoPhaseMotor
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import ReceivingTemporalMotorExpert


def _actor() -> ReceivingBandRefinedPrecontactMotor:
    base = _lateral_actor()
    values = {field.name: getattr(base, field.name) for field in fields(base) if field.init}
    values["maximum_relative_lateral_feature"] = 0.7696218896207085
    values["activation_states"] = ((0.0, 0.7666, 0.0, -0.472) + (0.0,) * 6,) * 3
    weights = base.candidate_neural_weights
    values["refined_neural_weights"] = replace(
        weights,
        output_bias=tuple(value + 0.01 for value in weights.output_bias),
    )
    return ReceivingBandRefinedPrecontactMotor(**values)


def test_refinement_is_measured_once_and_old_actor_is_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    actor = _actor()
    actor.propose(SimpleNamespace(frame=19, lateral_feature=0.7666, speed_feature=-0.472))
    assert actor.activation_enabled and actor.refined_episode
    assert actor.neural_weights == actor.refined_neural_weights
    actor.propose(SimpleNamespace(frame=20, lateral_feature=0.750, speed_feature=-0.40))
    assert actor.refined_episode

    old = _actor()
    old.propose(SimpleNamespace(frame=19, lateral_feature=0.754, speed_feature=-0.468))
    assert old.activation_enabled and not old.refined_episode
    assert old.neural_weights == old.candidate_neural_weights

    outside = _actor()
    parent = outside.neural_weights
    outside.propose(SimpleNamespace(frame=19, lateral_feature=0.780, speed_feature=-0.474))
    assert not outside.activation_enabled and not outside.refined_episode
    assert outside.neural_weights == parent


@pytest.mark.parametrize(
    "field,value",
    [
        ("refined_lateral_minimum", 0.770),
        ("refined_lateral_maximum", 0.776),
        ("refined_relative_vx_minimum", -0.56),
        ("refined_relative_vx_maximum", float("nan")),
        ("refined_lateral_minimum", True),
    ],
)
def test_invalid_refinement_bands_fail_closed(field: str, value: float) -> None:
    actor = _actor()
    setattr(actor, field, value)
    with pytest.raises(ValueError, match="bounded finite measured refinement band"):
        actor.__post_init__()
