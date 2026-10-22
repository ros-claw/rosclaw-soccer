"""Synthetic stream/codec tests only, NOT long-horizon native qualification."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_g1_velocity_policy import parameters, state

from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
from rosclaw_soccer.providers.g1.velocity_policy import G1VelocityPolicy, phase_delta


def fake_policy():
    policy = object.__new__(G1VelocityPolicy)
    policy.params = parameters()
    policy.contract_hash = "sha256:" + "a" * 64
    policy.phase = phase_delta()
    policy.last_action = np.zeros(29, np.float32)
    policy.last_observation = np.zeros(98, np.float32)
    policy.next_frame = policy.neural_calls = policy.attempted_neural_calls = 0
    policy.faulted = False
    policy.session = SimpleNamespace(
        run=lambda names, feed: [
            (
                feed["obs"][:, 69:] * np.float32(0.1) + feed["obs"][:, 11:40] * np.float32(0.2)
            ).astype(np.float32)
        ]
    )
    return policy


def stream(monkeypatch, horizon=600):
    monkeypatch.setattr(
        "rosclaw_soccer.providers.g1.continuous_velocity_stream.G1VelocityPolicy",
        lambda *args, **kwargs: fake_policy(),
    )
    return ContinuousVelocityStream(
        None,
        None,
        expected_model_hash="mock",
        expected_config_hash="mock",
        horizon_controls=horizon,
    )


def test_first300_exact_synthetic_codec_then_continues_without_phase_or_action_reset(monkeypatch):
    new, old = stream(monkeypatch), fake_policy()
    q, v = state()
    q[7:36] = np.linspace(-0.1, 0.1, 29)
    command = np.array([0.8, 0.2, 0.1], np.float32)
    for i in range(300):
        actual = new.propose(q, v, np.zeros(3), command, frame=i)
        expected = old.propose(q, v, np.zeros(3), command, frame=i)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(new.last_observation, old.last_observation)
        assert new.phase == old.phase
    phase, action = new.phase, new.last_action.copy()
    with pytest.raises(ValueError, match="H300"):
        old.propose(q, v, np.zeros(3), command, frame=300)
    new.propose(q, v, np.zeros(3), command, frame=300)
    np.testing.assert_array_equal(new.last_observation[69:], action)
    assert new.phase != phase_delta() and phase != new.phase
    for i in range(301, 600):
        new.propose(q, v, np.zeros(3), command, frame=i)
    assert new.neural_calls == new.next_frame == 600
    with pytest.raises(ValueError):
        new.propose(q, v, np.zeros(3), command, frame=600)
    with pytest.raises(ValueError, match="latched"):
        new.propose(q, v, np.zeros(3), command, frame=600)
    assert new.neural_calls == 600


@pytest.mark.parametrize("horizon", [True, 300, 6001, -1])
def test_no_unbounded_or_implicit_legacy_extension(monkeypatch, horizon):
    with pytest.raises(ValueError):
        stream(monkeypatch, horizon)


def test_two_players_have_private_history_and_fault_is_not_retry(monkeypatch):
    a, b = stream(monkeypatch), stream(monkeypatch)
    q, v = state()
    a.propose(q, v, np.zeros(3), np.zeros(3, np.float32), frame=0)
    assert b.next_frame == 0 and b.phase == phase_delta()
    assert a.last_action is not b.last_action
    with pytest.raises(ValueError):
        a.propose(q, v, np.zeros(3), np.zeros(3, np.float32), frame=0)
    assert a.neural_calls == 1 and a.faulted
