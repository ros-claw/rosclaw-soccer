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
        self._history = []
        self.reference = np.zeros((710, 36))

    def reset(self, state):
        self.resets += 1
        self._history.append((state.qpos.copy(), state.qvel.copy()))

    def observe(self, state):
        self.observations += 1

    def navigation_tick(self, state, frame):
        self.updates += 1
        return np.zeros(29)


def controller(monkeypatch, config=None):
    monkeypatch.setattr("rosclaw_soccer.providers.g1.sonic_navigation._StreamingBackend", Backend)
    return G1SonicNavigation(None, "red.defender", config)


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


def test_explicit_measured_start_uses_global_clock_without_duplicate_reset(monkeypatch):
    motor = controller(monkeypatch)
    current = observation(285)
    receipt = motor.start_from_observation(current)
    assert receipt.frame == 285 and receipt.time_sec == 5.7
    assert receipt.agent_id == current.agent_id and receipt.activation_ceiling == "SIM_ONLY"
    assert receipt.observation_hash.startswith("sha256:")
    assert motor.backend.resets == 1 and motor.backend.updates == 0
    motor.propose(current)
    motor.propose(observation(286))
    assert (motor.backend.resets, motor.backend.observations, motor.backend.updates) == (1, 1, 2)
    np.testing.assert_array_equal(motor.backend._history[0][0], current.qpos)
    assert motor._origin_frame == 285 and motor._next_frame == 287


def test_measured_start_cannot_skip_boundary_or_restart(monkeypatch):
    motor = controller(monkeypatch)
    motor.start_from_observation(observation(285))
    with pytest.raises(ValueError, match="latched off"):
        motor.propose(observation(286))
    with pytest.raises(ValueError, match="faulted"):
        motor.start_from_observation(observation(286))
    assert motor.backend.resets == 1 and motor.backend.updates == 0


@pytest.mark.parametrize("frame", [0, 285])
def test_duplicate_measured_start_latches_even_before_first_proposal(monkeypatch, frame):
    motor = controller(monkeypatch)
    motor.start_from_observation(observation(frame))
    with pytest.raises(ValueError, match="latched off"):
        motor.start_from_observation(observation(frame))
    with pytest.raises(ValueError, match="latched"):
        motor.propose(observation(frame))
    assert motor.backend.resets == 1


@pytest.mark.parametrize(
    "change",
    [
        {"agent_id": "blue.defender"},
        {"time_sec": 5.71},
        {"navigation_command": None},
        {"qpos": (0.0,) * 43},
    ],
)
def test_invalid_measured_start_latches_before_backend_reset(monkeypatch, change):
    motor = controller(monkeypatch)
    with pytest.raises(ValueError, match="latched off"):
        motor.start_from_observation(replace(observation(285), **change))
    assert motor.backend.resets == 0
    with pytest.raises(ValueError, match="faulted"):
        motor.start_from_observation(observation(285))


def test_used_or_retired_navigation_cannot_be_reinitialized(monkeypatch):
    motor = controller(monkeypatch)
    motor.propose(observation())
    with pytest.raises(ValueError, match="latched off"):
        motor.start_from_observation(observation(285))
    assert motor.backend.resets == 1
    retired = controller(monkeypatch)
    retired._retired = True
    with pytest.raises(ValueError, match="retired"):
        retired.start_from_observation(observation(285))
    assert retired.backend.resets == 0


def test_measured_start_backend_failure_is_terminal(monkeypatch):
    motor = controller(monkeypatch)
    motor.backend.reset = lambda state: (_ for _ in ()).throw(RuntimeError("planner failure"))
    with pytest.raises(ValueError, match="latched off"):
        motor.start_from_observation(observation(285))
    with pytest.raises(ValueError, match="faulted"):
        motor.start_from_observation(observation(285))


@pytest.mark.parametrize(
    "change",
    [
        {"navigation_command": (0.4, 0.0, 0.0)},
        {"target_position_m": (4.0, 0.0, 0.0)},
        {"intent": "shoot"},
        {"qvel": (0.01,) + (0.0,) * 40},
    ],
)
def test_measured_boundary_observation_cannot_change_before_proposal(monkeypatch, change):
    motor = controller(monkeypatch)
    current = observation(285)
    motor.start_from_observation(current)
    with pytest.raises(ValueError, match="latched off"):
        motor.propose(replace(current, **change))
    assert motor.backend.resets == 1 and motor.backend.updates == 0


def test_explicit_start_horizon_is_bounded_from_its_origin(monkeypatch):
    motor = controller(monkeypatch, SonicNavigationConfig(maximum_frames=50))
    motor.start_from_observation(observation(285))
    for frame in range(285, 335):
        motor.propose(observation(frame))
    with pytest.raises(ValueError, match="latched off"):
        motor.propose(observation(335))
    assert motor.backend.updates == 50 and motor.backend.resets == 1


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
