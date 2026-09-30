"""Measured speed gate rejects unsafe fast near-side activation."""

from dataclasses import fields
from types import SimpleNamespace

import pytest
from test_receiving_gated_precontact_motor import _actor as _gated_actor

from rosclaw_soccer.rsi.receiving_gated_precontact_motor import ReceivingGatedPrecontactMotor
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import ReceivingTemporalMotorExpert
from rosclaw_soccer.rsi.receiving_velocity_gated_precontact_motor import (
    ReceivingVelocityGatedPrecontactMotor,
)


def _actor() -> ReceivingVelocityGatedPrecontactMotor:
    base = _gated_actor()
    values = {field.name: getattr(base, field.name) for field in fields(base) if field.init}
    values["activation_states"] = ((0.0,) * 3 + (-0.48,) + (0.0,) * 6,) * 3
    return ReceivingVelocityGatedPrecontactMotor(**values)


def test_speed_gate_is_first_frame_measured_and_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ReceivingGatedPrecontactMotor,
        "propose",
        lambda self, observation: (0.0,) * 29,
    )
    monkeypatch.setattr(
        ReceivingTemporalMotorExpert,
        "features",
        staticmethod(lambda observation: (0.0,) * 3 + (observation.speed_feature,) + (0.0,) * 6),
    )
    slow = _actor()
    slow.propose(SimpleNamespace(frame=19, speed_feature=-0.47))
    assert slow.activation_enabled
    assert slow.neural_weights == slow.candidate_neural_weights
    slow.propose(SimpleNamespace(frame=20, speed_feature=-0.50))
    assert slow.activation_enabled
    fast = _actor()
    parent_weights = fast.neural_weights
    fast.propose(SimpleNamespace(frame=19, speed_feature=-0.485))
    assert fast.activation_decided and not fast.activation_enabled
    assert fast.neural_weights == parent_weights


def test_speed_threshold_rejects_nonfinite() -> None:
    actor = _actor()
    actor.minimum_relative_vx_feature = float("nan")
    with pytest.raises(ValueError, match="bounded measured"):
        actor.__post_init__()
