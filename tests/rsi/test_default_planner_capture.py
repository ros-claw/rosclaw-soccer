"""Observation-only reference recording, never a physical qualification."""

import argparse
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_navigation import SonicNavigationConfig, _StreamingBackend
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode
from rosclaw_soccer.rsi.postcontact_navigation_probe import PostcontactNavigationProbe


class Planner:
    def __init__(self):
        self.feeds = []

    def run(self, names, feed):
        self.feeds.append(copy.deepcopy(feed))
        q = np.repeat(feed["context_mujoco_qpos"][:, :1], 16, axis=1)
        return q, np.array([16], dtype=np.int64)


@pytest.mark.parametrize("speed", [0.0, 0.02, 0.8, 1.4])
def test_capture_does_not_change_any_planner_feed_or_numerical_reference(speed):
    outputs, backends = [], []
    q = np.zeros((4, 36))
    q[:, 3] = 1
    for capture in (False, True):
        backend = object.__new__(_StreamingBackend)
        backend.navigation = SonicNavigationConfig(
            model_variant="low_latency", record_default_planner_events=capture
        )
        backend.command = (speed, 0.0, 0.0)
        backend.facing = 0.0
        backend.planner_calls = 0
        backend.events = []
        backend._planner = Planner()
        outputs.append(backend.plan(q, 0))
        backends.append(backend)
    np.testing.assert_array_equal(*outputs)
    a, b = [backend._planner.feeds[0] for backend in backends]
    assert set(a) == set(b)
    for key in a:
        assert a[key].dtype == b[key].dtype
        np.testing.assert_array_equal(a[key], b[key])
    event = backends[1].events[0]
    assert event["planner_mode"] == int(b["mode"][0])
    assert {key: value for key, value in event.items() if key != "planner_mode"} == backends[
        0
    ].events[0]


@pytest.mark.parametrize("capture", [None, 1, "yes"])
def test_recording_requires_explicit_boolean(capture):
    with pytest.raises(ValueError):
        SonicNavigationConfig(model_variant="low_latency", record_default_planner_events=capture)


def test_capture_cannot_hide_a_mode_intervention():
    with pytest.raises(ValueError):
        SonicNavigationConfig(
            model_variant="low_latency",
            experimental_maximum_speed_mps=1.5,
            experimental_locomotion_mode=3,
            record_default_planner_events=True,
        )


@pytest.mark.parametrize("kind", ["parent", "bounded", "postcontact"])
def test_default_capture_accepts_complete_catalog_control_shapes_before_allocation(
    tmp_path, monkeypatch, kind
):
    import sys

    def unopened(path):
        raise RuntimeError("allocator sentinel: no physical allocation")

    monkeypatch.setitem(
        sys.modules, "mujoco", SimpleNamespace(MjSpec=SimpleNamespace(from_file=unopened))
    )
    profile = None if kind == "parent" else BoundedFirstTouchControl()
    with pytest.raises(RuntimeError, match="allocator sentinel"):
        run_native_first_touch_episode(
            argparse.Namespace(scene=tmp_path / "absent.xml", foundation_only=False),
            course=(2.3, 0.0, -0.4),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            control_profile=profile,
            approach_mode="legacy" if profile is None else "bounded_approach_diagnostic",
            swing_mode="legacy" if profile is None else "bounded_contact_diagnostic",
            postcontact_profile=PostcontactNavigationProbe() if kind == "postcontact" else None,
            recurrent_sampling_factory=object(),
            record_default_planner_events=True,
        )
    assert not list(tmp_path.iterdir())


def test_default_capture_contract_is_opt_in(monkeypatch):
    from rosclaw_soccer.providers.g1 import sonic_navigation as nav

    class Backend:
        def __init__(self, *args):
            self.qualification = SimpleNamespace(qualification_hash="sha256:" + "a" * 64)

    monkeypatch.setattr(nav, "_StreamingBackend", Backend)
    base = SonicNavigationConfig(model_variant="low_latency")
    original = nav.G1SonicNavigation(None, "test.player", base)
    explicit_false = nav.G1SonicNavigation(
        None, "test.player", replace(base, record_default_planner_events=False)
    )
    captured = nav.G1SonicNavigation(
        None, "test.player", replace(base, record_default_planner_events=True)
    )
    assert original.contract_hash == explicit_false.contract_hash
    assert captured.contract_hash != original.contract_hash
