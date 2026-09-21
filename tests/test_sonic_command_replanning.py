from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_command_replanning import command_event_requires_replan
from rosclaw_soccer.providers.g1.sonic_navigation import (
    SonicNavigationConfig,
    _StreamingBackend,
)
from rosclaw_soccer.sim.contracts import hash_json


def due(**kwargs):
    values = dict(
        frame=5,
        last_plan_frame=0,
        command=(0.0, 0.0, 0.0),
        last_planned_command=(0.7, 0.0, 0.0),
    )
    return command_event_requires_replan(**(values | kwargs))


@pytest.mark.parametrize("frame,expected", [(0, False), (4, False), (5, True), (20, True)])
def test_event_has_minimum_interval(frame, expected):
    assert due(frame=frame) is expected


@pytest.mark.parametrize(
    "command,expected",
    [
        ((0.0, 0.0, 0.0), False),
        ((0.149, 0.0, 0.0), False),
        ((0.15, 0.0, 0.0), True),
        ((0.0, -0.15, 0.0), True),
        ((0.0, 0.0, 0.299), False),
        ((0.0, 0.0, 0.3), True),
        ((0.0, 0.0, -0.3), True),
    ],
)
def test_fixed_thresholds(command, expected):
    assert due(command=command, last_planned_command=(0.0, 0.0, 0.0)) is expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame": True},
        {"frame": -1},
        {"frame": 3001},
        {"last_plan_frame": False},
        {"last_plan_frame": -1},
        {"last_plan_frame": 6},
        {"command": [0.0, 0.0, 0.0]},
        {"command": (0.0, 0.0)},
        {"command": (True, 0.0, 0.0)},
        {"command": (float("nan"), 0.0, 0.0)},
        {"command": (0.0, float("inf"), 0.0)},
        {"command": (0.5, 0.5, 0.0)},
        {"command": (0.0, 0.0, 1.51)},
        {"last_planned_command": (0.8, 0.0, 0.0)},
    ],
)
def test_bad_context_rejected_even_during_cooldown(kwargs):
    with pytest.raises(ValueError):
        due(**kwargs)
    with pytest.raises(ValueError):
        due(**({"frame": 0} | kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"experimental_command_replanning": 1},
        {"experimental_command_replanning": "true"},
        {"experimental_command_replanning": True, "model_variant": "sonic_v1_1"},
        {"experimental_command_replanning": True, "experimental_maximum_speed_mps": 1.0},
        {"experimental_command_replanning": True, "latent_schedule": object()},
        {"experimental_command_replanning": True, "pose_reference": object()},
    ],
)
def test_opt_in_must_not_mix_unqualified_variants(kwargs):
    with pytest.raises(ValueError):
        SonicNavigationConfig(**({"model_variant": "low_latency"} | kwargs))


def backend(*, enabled):
    obj = object.__new__(_StreamingBackend)
    obj.navigation = SonicNavigationConfig(
        model_variant="low_latency", experimental_command_replanning=enabled
    )
    obj.command = (0.7, 0.0, 0.0)
    obj.last_planned_command = obj.command
    obj.last_planned_frame = 0
    obj.reference = np.zeros((710, 36))
    obj.reference[:, 3] = 1.0
    obj.calls = []

    def plan(context, frame):
        obj.calls.append((frame, context.copy()))
        obj.last_planned_frame = frame
        obj.last_planned_command = tuple(obj.command)
        segment = np.zeros((40, 36))
        segment[:, 0] = 1.0
        segment[:, 3] = 1.0
        return segment

    obj.plan = plan
    obj._update_from_reference = lambda state, frame: np.zeros(29)
    return obj


def test_default_cadence_ignores_event_and_keeps_original_splice():
    obj = backend(enabled=False)
    obj.command = (0.0, 0.0, 0.0)
    for frame in range(21):
        obj.navigation_tick(SimpleNamespace(), frame)
    assert [frame for frame, _ in obj.calls] == [20]
    assert np.all(obj.reference[:30, 0] == 0)
    assert np.all(obj.reference[30:, 0] == 1)


def test_event_uses_future_splice_and_keeps_policy_history_untouched():
    obj = backend(enabled=True)
    obj._history = ["do-not-reset"]
    original = obj.reference.copy()
    obj.command = (0.0, 0.0, 0.0)
    obj.navigation_tick(SimpleNamespace(), 5)
    assert [frame for frame, _ in obj.calls] == [5]
    np.testing.assert_array_equal(obj.reference[:9], original[:9])
    assert np.all(obj.reference[9:, 0] == 1)
    assert obj._history == ["do-not-reset"]
    obj.navigation_tick(SimpleNamespace(), 6)
    assert len(obj.calls) == 1
    obj.command = (0.0, 0.5, 0.0)
    obj.navigation_tick(SimpleNamespace(), 9)
    assert len(obj.calls) == 1
    obj.navigation_tick(SimpleNamespace(), 10)
    assert [frame for frame, _ in obj.calls] == [5, 10]


def test_incremental_changes_accumulate_since_successful_plan():
    obj = backend(enabled=True)
    for frame, speed in ((5, 0.65), (6, 0.60), (7, 0.50)):
        obj.command = (speed, 0.0, 0.0)
        obj.navigation_tick(SimpleNamespace(), frame)
    assert [frame for frame, _ in obj.calls] == [7]


def test_event_and_periodic_tick_plan_only_once():
    obj = backend(enabled=True)
    obj.command = (0.0, 0.0, 0.0)
    obj.navigation_tick(SimpleNamespace(), 20)
    assert [frame for frame, _ in obj.calls] == [20]


def test_legacy_contract_hash_unchanged_and_experiment_is_bound(monkeypatch):
    from test_s304_sonic_navigation import controller

    motor = controller(monkeypatch, SonicNavigationConfig(model_variant="low_latency"))
    config = asdict(motor.config)
    for key in (
        "pose_reference",
        "latent_schedule",
        "experimental_maximum_speed_mps",
        "experimental_command_replanning",
    ):
        config.pop(key)
    expected = hash_json(
        dict(
            schema="rosclaw_soccer.g1_sonic_navigation.v1",
            agent_id="red.defender",
            config=config,
            foundation="sha256:" + "a" * 64,
            command="post-clearance world vx vy <= .7 m/s, yaw rate <= 1.5 rad/s",
            activation_ceiling="SIM_ONLY",
        )
    )
    assert motor.contract_hash == expected
    experimental = controller(
        monkeypatch,
        SonicNavigationConfig(model_variant="low_latency", experimental_command_replanning=True),
    )
    assert experimental.contract_hash != motor.contract_hash


def test_receiving_option_passes_only_explicit_opt_in(monkeypatch):
    from test_receiving_sonic import motor

    assert motor(monkeypatch).navigation.config.experimental_command_replanning is False
    assert (
        motor(
            monkeypatch, experimental_command_replanning=True
        ).navigation.config.experimental_command_replanning
        is True
    )
