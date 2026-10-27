from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
from test_receiving_smooth_residual_motor import _actor

from rosclaw_soccer.rsi.receiving_foot_servo_phase_motor import ReceivingFootServoPhaseMotor
from rosclaw_soccer.rsi.receiving_temporal_motor_expert import ReceivingTemporalMotorExpert
from rosclaw_soccer.rsi.team_receiving_selector import TeamReceivingSelector
from rosclaw_soccer.rsi.team_receiving_selector_motor import TeamReceivingSelectorMotor


def make_motor(action):
    parent = _actor()
    values = {f.name: getattr(parent, f.name) for f in fields(parent) if f.init}
    biases = np.zeros(16)
    biases[action] = 1
    offsets = np.zeros((16, 12))
    offsets[1] = 0.1
    selector = TeamReceivingSelector.from_dict(
        dict(
            schema="soccer.team_receiving_offline_actor.v1",
            hardware_authorized=False,
            promotion_authorized=False,
            center=[0.0] * 10,
            scale=[1.0] * 10,
            weight1=np.zeros((32, 10)).tolist(),
            bias1=[0.0] * 32,
            weight2=np.zeros((16, 32)).tolist(),
            bias2=biases.tolist(),
            offsets=offsets.tolist(),
        )
    )
    return TeamReceivingSelectorMotor(**values, selector=selector)


def test_parent_fallback_and_learned_action_respect_activation(monkeypatch):
    monkeypatch.setattr(ReceivingFootServoPhaseMotor, "propose", lambda self, obs: (0.0,) * 29)
    monkeypatch.setattr(
        ReceivingTemporalMotorExpert,
        "features",
        staticmethod(lambda obs: (0.0, obs.lateral, 0.0, obs.speed, *(0.0,) * 6)),
    )
    observation = SimpleNamespace(frame=19, lateral=0.766, speed=-0.45)
    parent = _actor()
    parent.propose(observation)
    fallback = make_motor(0)
    fallback.propose(observation)
    assert fallback.neural_weights == parent.neural_weights
    learned = make_motor(1)
    learned.propose(observation)
    assert learned.selected_action == 1
    assert learned.neural_weights.output_bias == pytest.approx(
        np.asarray(learned.candidate_neural_weights.output_bias) + 0.1
    )
    blocked = make_motor(1)
    blocked.propose(SimpleNamespace(frame=19, lateral=0.8, speed=-0.45))
    assert not blocked.activation_enabled
    assert blocked.selected_action == 0

    protected = make_motor(1)
    protected.protected_episode = True
    protected.propose(observation)
    assert protected.selected_action == 0
    assert protected.neural_weights == parent.neural_weights
