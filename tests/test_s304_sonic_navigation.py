from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_navigation import (
    G1SonicNavigation,
    SonicNavigationConfig,
    _StreamingBackend,
    future_reference_context,
)
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation


def observation(frame=0):
    q = np.zeros(43)
    q[2] = 0.78
    q[3] = q[39] = 1.0
    return TeamMotorObservation(
        "red.defender",
        frame,
        frame * 0.02,
        "other",
        False,
        tuple(q.tolist()),
        (0.0,) * 41,
        (3.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
    )


@pytest.mark.parametrize(
    "command",
    [
        (0.71, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.0, 0.0, 1.51),
        [0.0] * 3,
        (True, 0.0, 0.0),
        (0.0, float("nan"), 0.0),
    ],
)
def test_navigation_command_is_immutable_finite_and_no_larger_than_world(command):
    with pytest.raises(ValueError):
        replace(observation(), navigation_command=command)


@pytest.mark.parametrize(
    "config",
    [
        {"maximum_frames": True},
        {"maximum_frames": 3001},
        {"replan_frames": 10},
        {"lookahead_frames": 9},
        {"planner_seed": -1},
    ],
)
def test_navigation_has_explicit_bounded_horizon(config):
    with pytest.raises(ValueError):
        SonicNavigationConfig(**config)


def test_context_keeps_quaternion_hemisphere_without_mutating_reference():
    ref = np.zeros((100, 36))
    ref[:, 3] = 1.0
    ref[::2, 3] = -1.0
    ref[:, 0] = np.arange(100)
    saved = ref.copy()
    result = future_reference_context(ref, 10)
    np.testing.assert_allclose(result[:, 0], 10 + np.arange(4) * 50 / 30)
    np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=1), 1)
    np.testing.assert_array_equal(ref, saved)
    with pytest.raises(ValueError):
        future_reference_context(ref, 99)
    ref[10:20, 3:7] = 0
    with pytest.raises(ValueError):
        future_reference_context(ref, 10)


class Backend:
    def __init__(self, *args):
        self.qualification = SimpleNamespace(qualification_hash="sha256:" + "a" * 64)
        self.kp = np.ones(29) * 50
        self.kd = np.ones(29)
        self.resets = self.observations = self.updates = 0

    def reset(self, state):
        self.resets += 1

    def observe(self, state):
        self.observations += 1

    def navigation_tick(self, state, frame):
        self.updates += 1
        return np.zeros(29)


def controller(monkeypatch):
    monkeypatch.setattr("rosclaw_soccer.providers.g1.sonic_navigation._StreamingBackend", Backend)
    return G1SonicNavigation(None, "red.defender")


def test_navigation_history_continuous_and_fault_never_silently_restarts(monkeypatch):
    motor = controller(monkeypatch)
    for frame in range(21):
        motor.propose(observation(frame))
    assert (motor.backend.resets, motor.backend.observations, motor.backend.updates) == (1, 20, 21)
    assert motor.backend.command == (0.5, 0.0, 0.0)
    with pytest.raises(ValueError):
        motor.propose(observation(20))
    with pytest.raises(ValueError, match="latched"):
        motor.propose(observation(21))
    assert motor.backend.updates == 21


@pytest.mark.parametrize(
    "change",
    [
        {"navigation_command": None},
        {"agent_id": "blue.defender"},
        {"time_sec": 0.01},
        {"frame": 1},
        {"qpos": (0.0,) * 43},
    ],
)
def test_missing_or_foreign_input_latches_before_inference(monkeypatch, change):
    motor = controller(monkeypatch)
    with pytest.raises(ValueError):
        motor.propose(replace(observation(), **change))
    assert motor.backend.updates == motor.backend.resets == 0
    with pytest.raises(ValueError):
        motor.propose(observation())


def test_backend_failure_is_converted_to_shared_world_fault(monkeypatch):
    motor = controller(monkeypatch)
    motor.backend.navigation_tick = lambda *args: (_ for _ in ()).throw(RuntimeError("inference"))
    with pytest.raises(ValueError, match="latched off"):
        motor.propose(observation())
    with pytest.raises(ValueError, match="latched"):
        motor.propose(observation())


@pytest.mark.parametrize("fault", ["shape", "count", "nan", "quaternion"])
def test_malformed_planner_output_rejected(fault):
    backend = object.__new__(_StreamingBackend)
    backend.navigation = SonicNavigationConfig()
    backend.command = (0.5, 0.0, 0.0)
    backend.facing = 0.0
    backend.planner_calls = 0
    backend.events = []
    output = np.zeros((1, 10, 36))
    output[:, :, 3] = 1.0
    count = np.asarray([10], dtype=np.int64)
    if fault == "shape":
        output = output[0]
    elif fault == "count":
        count = np.asarray([10.5])
    elif fault == "nan":
        output[0, 0, 0] = np.nan
    else:
        output[:, :, 3] = 0.0
    backend._planner = SimpleNamespace(run=lambda *args: (output, count))
    with pytest.raises(ValueError):
        backend.plan(np.zeros((4, 36)), 0)
    assert not backend.events and backend.planner_calls == 0
