import argparse
import copy
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_navigation import SonicNavigationConfig, _StreamingBackend
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode
from rosclaw_soccer.rsi.sonic_encoder_reference_audit import (
    audit_encoder_reference_consumption,
    low_latency_legacy_features,
)
from rosclaw_soccer.rsi.sonic_locomotion_reference_audit import replay_locomotion_segments
from tests.rsi.test_sonic_locomotion_reference_audit import Planner


def config(height):
    return SonicNavigationConfig(
        maximum_frames=300,
        model_variant="low_latency",
        experimental_maximum_speed_mps=1.5,
        experimental_locomotion_mode=3,
        experimental_reference_height_m=height,
    )


class HeightPlanner(Planner):
    def __init__(self):
        super().__init__()
        self.heights = []

    def run(self, names, feed):
        self.heights.append(float(feed["height"][0]))
        output, count = super().run(names, feed)
        if feed["height"][0] >= 0:
            output[:, :, 7] += feed["height"][0] / 10
        return output, count


def fixture():
    q = np.zeros((300, 1, 43))
    q[:, 0, 2] = 0.793
    q[:, 0, 3] = 1
    command = np.zeros((300, 1, 3))
    command[:, 0, 0] = 1.4
    command[60, 0, 0] = 0.8
    command[80, 0, 0] = 0
    backend = object.__new__(_StreamingBackend)
    backend.navigation = config(0.75)
    backend._planner = HeightPlanner()
    backend.planner_calls = 0
    backend.events = []
    backend.facing = 0.0
    backend.command = tuple(command[0, 0])
    backend.reference = backend._generate_reference(q[0, 0, :36])
    for frame in range(20, 300, 20):
        backend.command = tuple(command[frame, 0])
        backend.refresh_reference(None, frame)
    return dict(canonical_qpos=q, navigation_command=command), backend


@pytest.mark.parametrize("height", [None, 0.65, 0.7, 0.75, 0.8])
def test_explicit_bounded_height_configuration(height):
    assert config(height).experimental_reference_height_m == height


@pytest.mark.parametrize("height", [True, 1, "0.75", float("nan"), float("inf"), 0.649, 0.801])
def test_invalid_height_rejected_before_allocation(height):
    with pytest.raises(ValueError):
        config(height)


def test_height_cannot_be_attached_to_walk_or_default_configuration():
    with pytest.raises(ValueError):
        SonicNavigationConfig(experimental_reference_height_m=0.75)
    with pytest.raises(ValueError):
        SonicNavigationConfig(
            model_variant="low_latency",
            experimental_maximum_speed_mps=1.5,
            experimental_locomotion_mode=2,
            experimental_reference_height_m=0.75,
        )


def test_producer_and_independent_replayer_bind_height_and_preserve_low_modes():
    trace, backend = fixture()
    session = HeightPlanner()
    rows = replay_locomotion_segments(
        session, trace, backend.events, 3, reference_height_probe_m=0.75
    )
    assert len(rows) == session.calls == 15
    assert backend._planner.heights == session.heights
    assert session.heights[3:5] == [-1.0, -1.0]
    assert all(
        e["reference_height_input_m"] == h
        for e, h in zip(backend.events, session.heights, strict=True)
    )


@pytest.mark.parametrize(
    "fault", ["missing", "changed", "boolean", "undeclared", "wrong_declared_height"]
)
def test_changed_height_or_undeclared_events_fail_before_model_calls(fault):
    trace, backend = fixture()
    events = copy.deepcopy(backend.events)
    height = 0.75
    if fault == "missing":
        del events[0]["reference_height_input_m"]
    elif fault == "changed":
        events[3]["reference_height_input_m"] = 0.75
    elif fault == "boolean":
        events[0]["reference_height_input_m"] = True
    elif fault == "undeclared":
        height = None
    else:
        height = 0.7
    session = HeightPlanner()
    with pytest.raises(ValueError):
        replay_locomotion_segments(session, trace, events, 3, reference_height_probe_m=height)
    assert session.calls == 0


def test_actual_encoder_comparison_consumes_the_declared_height_reference():
    trace, backend = fixture()
    features = np.empty((300, 1, 640), dtype=np.float32)

    def observe(frame, reference):
        for tick in range(frame, frame + 20):
            features[tick, 0] = low_latency_legacy_features(
                reference[tick : tick + 10], trace["canonical_qpos"][tick, 0, 3:7]
            )

    replay_locomotion_segments(
        HeightPlanner(),
        trace,
        backend.events,
        3,
        reference_observer=observe,
        reference_height_probe_m=0.75,
    )
    trace["foundation_neural_encoder_features"] = features
    result = audit_encoder_reference_consumption(
        HeightPlanner(), trace, backend.events, 3, reference_height_probe_m=0.75
    )
    assert result["maximum_absolute_error"] == 0
    assert result["high_speed_reference_height_m"] == 0.75


@pytest.mark.parametrize(
    "probe,mode,height",
    [(False, 3, 0.75), (True, 2, 0.75), (True, 3, True), (True, 3, float("nan")), (True, 3, 0.9)],
)
def test_native_height_rejects_mixed_or_invalid_modes_before_allocation(probe, mode, height):
    with pytest.raises(ValueError, match="reference-height"):
        run_native_first_touch_episode(
            argparse.Namespace(),
            course=(2.5, 0.0, -0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            foundation_approach_probe=probe,
            locomotion_mode_probe=mode,
            reference_height_probe_m=height,
        )


def test_valid_captured_native_height_reaches_allocator_without_residuals(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    def unopened(path):
        raise RuntimeError("allocator boundary")

    monkeypatch.setitem(
        sys.modules, "mujoco", SimpleNamespace(MjSpec=SimpleNamespace(from_file=unopened))
    )
    with pytest.raises(RuntimeError, match="allocator boundary"):
        run_native_first_touch_episode(
            argparse.Namespace(
                foundation_only=True,
                record_foundation_observation=True,
                scene=tmp_path / "missing.xml",
            ),
            course=(2.5, 0.0, -0.5),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            foundation_approach_probe=True,
            locomotion_mode_probe=3,
            reference_height_probe_m=0.75,
            control_profile=BoundedFirstTouchControl(),
            approach_mode="bounded_approach_diagnostic",
        )
