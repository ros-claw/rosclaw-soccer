from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
from test_receiving_smooth_residual_motor import _actor
from test_team_coverage_actor import model

from rosclaw_soccer.rsi.receiving_foot_servo_phase_motor import ReceivingFootServoPhaseMotor
from rosclaw_soccer.rsi.receiving_kinematic_temporal_expert import ReceivingKinematicTemporalExpert
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import ReceivingTemporalMotorExpert
from rosclaw_soccer.rsi.team_coverage_actor import TeamCoverageActor
from rosclaw_soccer.rsi.team_coverage_motor import TeamCoverageMotor


def test_expanded_actor_preserves_parent_and_protection(monkeypatch):
    monkeypatch.setattr(ReceivingFootServoPhaseMotor, "propose", lambda self, obs: (0.0,) * 29)
    monkeypatch.setattr(
        ReceivingKinematicTemporalExpert, "features", staticmethod(lambda obs: (0.0,) * 48)
    )
    monkeypatch.setattr(
        ReceivingTemporalMotorExpert,
        "features",
        staticmethod(lambda obs: (0.0, 0.766, 0.0, obs.speed, *(0.0,) * 6)),
    )
    parent = _actor()
    values = {f.name: getattr(parent, f.name) for f in fields(parent) if f.init}
    values["activation_states"] = ((0.0, 0.766, 0.0, -0.475, *(0.0,) * 6),) * 3
    values["activation_radius"] = 0.019
    value = model()
    value["support"] = [[0.0, 0.766, 0.0, -0.484, *([0.0] * 6)]] * 6
    value["offsets"][1] = [0.1] * 12
    actor = TeamCoverageActor.from_dict(value)
    motor = TeamCoverageMotor(**values, coverage_actor=actor)
    motor.propose(SimpleNamespace(frame=19, speed=-0.484))
    assert motor.coverage_action == 1
    assert motor.neural_weights.output_bias == pytest.approx(
        np.asarray(motor.candidate_neural_weights.output_bias) + 0.1
    )
    for speed, protected in [(-0.475, False), (-0.484, True), (-0.5, False)]:
        baseline = type(parent)(**values)
        candidate = TeamCoverageMotor(**values, coverage_actor=actor)
        baseline.protected_episode = candidate.protected_episode = protected
        observation = SimpleNamespace(frame=19, speed=speed)
        baseline.propose(observation)
        candidate.propose(observation)
        assert candidate.coverage_action == 16
        assert candidate.neural_weights == baseline.neural_weights
        assert candidate.activation_enabled == baseline.activation_enabled


def test_protection_is_decided_before_coverage_selection(monkeypatch):
    monkeypatch.setattr(ReceivingFootServoPhaseMotor, "propose", lambda self, obs: (0.0,) * 29)
    monkeypatch.setattr(
        ReceivingKinematicTemporalExpert,
        "features",
        staticmethod(lambda obs: (0.1,) * 10 + (0.0,) * 38),
    )
    monkeypatch.setattr(
        ReceivingTemporalMotorExpert,
        "features",
        staticmethod(lambda obs: (0.0, 0.766, 0.0, -0.484, *(0.0,) * 6)),
    )
    parent = _actor()
    values = {f.name: getattr(parent, f.name) for f in fields(parent) if f.init}
    values["activation_states"] = ((0.0, 0.766, 0.0, -0.475, *(0.0,) * 6),) * 3
    values["activation_radius"] = 0.019
    value = model()
    value["support"] = [[0.0, 0.766, 0.0, -0.484, *([0.0] * 6)]] * 6
    motor = TeamCoverageMotor(**values, coverage_actor=TeamCoverageActor.from_dict(value))
    assert motor.protected_episode is None
    motor.propose(SimpleNamespace(frame=19))
    assert motor.protected_episode is True
    assert motor.coverage_action == 16
