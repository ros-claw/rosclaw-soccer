"""Two measured coordinates distinguish near-side skill from old far-side body state."""

from dataclasses import fields
from types import SimpleNamespace

import pytest
from test_receiving_velocity_gated_precontact_motor import _actor as _velocity_actor

from rosclaw_soccer.rsi.receiving_lateral_velocity_gated_precontact_motor import (
    ReceivingLateralVelocityGatedPrecontactMotor,
)
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import ReceivingTemporalMotorExpert
from rosclaw_soccer.rsi.receiving_velocity_gated_precontact_motor import (
    ReceivingVelocityGatedPrecontactMotor,
)


def _actor() -> ReceivingLateralVelocityGatedPrecontactMotor:
    base = _velocity_actor()
    values = {field.name: getattr(base, field.name) for field in fields(base) if field.init}
    values["activation_states"] = ((0.0, 0.764, 0.0, -0.45) + (0.0,) * 6,) * 3
    values["activation_radius"] = 0.019
    return ReceivingLateralVelocityGatedPrecontactMotor(**values)


def test_lateral_boundary_preserves_old_state_without_future_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ReceivingVelocityGatedPrecontactMotor,
        "propose",
        lambda self, observation: (0.0,) * 29,
    )
    monkeypatch.setattr(
        ReceivingTemporalMotorExpert,
        "features",
        staticmethod(
            lambda observation: (
                (
                    0.0,
                    observation.lateral_feature,
                    0.0,
                    observation.speed_feature,
                )
                + (0.0,) * 6
            )
        ),
    )
    near = _actor()
    near.propose(SimpleNamespace(frame=19, lateral_feature=0.764, speed_feature=-0.45))
    assert near.activation_enabled
    near.propose(SimpleNamespace(frame=20, lateral_feature=0.80, speed_feature=-0.50))
    assert near.activation_enabled

    far = _actor()
    parent = far.neural_weights
    far.propose(SimpleNamespace(frame=19, lateral_feature=0.775, speed_feature=-0.45))
    assert far.activation_decided and not far.activation_enabled
    assert far.neural_weights == parent


def test_lateral_boundary_is_bounded_and_finite() -> None:
    actor = _actor()
    for invalid in (0.759, 0.776, float("nan"), True):
        actor.maximum_relative_lateral_feature = invalid
        with pytest.raises(ValueError, match="bounded measured lateral"):
            actor.__post_init__()
