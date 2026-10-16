import argparse
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_navigation import SonicNavigationConfig, _StreamingBackend
from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.fast_intercept_control import (
    FastInterceptControl,
    require_fast_navigation_trace,
)
from rosclaw_soccer.rsi.native_first_touch_episode import run_native_first_touch_episode
from rosclaw_soccer.rsi.sonic_locomotion_reference_audit import replay_locomotion_segments
from rosclaw_soccer.skills.team.navigation_envelope import (
    FastInterceptNavigationEnvelope,
    SimulationNavigationEnvelope,
)
from tests.rsi.test_sonic_reference_height_probe import HeightPlanner


def config():
    return SonicNavigationConfig(
        model_variant="low_latency",
        experimental_maximum_speed_mps=2.1,
        experimental_locomotion_mode=3,
        experimental_reference_height_m=0.75,
        experimental_fast_intercept=True,
    )


@pytest.mark.parametrize("first", [None, 0, 100, 299])
@pytest.mark.parametrize("body", [0, 4])
def test_only_precontact_cap_changes_and_independent_rule_matches(first, body):
    base, fast = BoundedFirstTouchControl(), FastInterceptControl()
    trace = dict(
        root_pose_xyzw_m=np.zeros((300, 1, 7)),
        ball_position_before_step_m=np.zeros((300, 1, 3)),
        ball_linear_velocity_before_step_m_s=np.zeros((300, 1, 3)),
        force_n=np.zeros((300, 1, 6)),
        navigation_command=np.zeros((300, 1, 3)),
        approach_requested_navigation_command=np.zeros((300, 1, 3)),
    )
    trace["root_pose_xyzw_m"][:, :, 6] = 1.0
    trace["ball_position_before_step_m"][:, :, 0] = 2.7
    trace["ball_position_before_step_m"][:, :, 1] = 0.15
    trace["ball_linear_velocity_before_step_m_s"][:, :, 0] = 0.6
    if first is not None:
        trace["force_n"][first, 0, body] = 2.0
    for frame in range(300):
        before = first is None or frame <= first
        kw = dict(gap_x=2.7, gap_y=0.15, ball_vx=0.6, before_contact=before)
        raw, actual = fast.navigation(**kw)
        old_raw, old = base.navigation(**kw)
        assert raw == old_raw and actual[1:] == old[1:]
        assert actual[0] == (2.0 if before else 1.4)
        trace["approach_requested_navigation_command"][frame, 0] = raw
        trace["navigation_command"][frame, 0] = actual
    require_fast_navigation_trace(trace)
    trace["navigation_command"][200, 0, 0] += 0.01
    with pytest.raises(ValueError, match="scalar rule"):
        require_fast_navigation_trace(trace)


@pytest.mark.parametrize(
    "kw",
    [
        dict(precontact_forward_cap_m_s=True),
        dict(precontact_forward_cap_m_s=1.8),
        dict(precontact_forward_cap_m_s=2),
        dict(precontact_forward_cap_m_s=float("nan")),
        dict(forward_cap_m_s=1.48),
        dict(closing_gain_s_inv=2.0),
        dict(foot_forward_cap_m=0.1),
    ],
)
def test_unregistered_factor_or_nonfloat_rejected(kw):
    with pytest.raises(ValueError):
        FastInterceptControl(**kw)


def test_original_navigation_envelope_is_not_broadened():
    with pytest.raises(ValueError):
        SimulationNavigationEnvelope("cpu.first_touch.0", "sha256:" + "a" * 64, 2.1)
    FastInterceptNavigationEnvelope("cpu.first_touch.0", "sha256:" + "a" * 64, 2.1)
    for speed in (1.5, 2.0, 2.2, True):
        with pytest.raises(ValueError):
            FastInterceptNavigationEnvelope("cpu.first_touch.0", "sha256:" + "a" * 64, speed)
    with pytest.raises(ValueError):
        FastInterceptNavigationEnvelope("cpu.first_touch.0", "sha256:" + "a" * 64, 2.1, "REAL")


def test_fast_config_is_explicit_and_cannot_weaken_default_range():
    assert config().experimental_fast_intercept
    with pytest.raises(ValueError):
        SonicNavigationConfig(
            model_variant="low_latency",
            experimental_maximum_speed_mps=2.1,
            experimental_locomotion_mode=3,
            experimental_reference_height_m=0.75,
        )
    for kw in (
        dict(experimental_fast_intercept=1),
        dict(experimental_fast_intercept=True),
        dict(
            experimental_maximum_speed_mps=2.1,
            experimental_locomotion_mode=3,
            experimental_reference_height_m=0.7,
            experimental_fast_intercept=True,
        ),
    ):
        with pytest.raises(ValueError):
            SonicNavigationConfig(**kw)


def test_fast_reference_replay_is_explicit_and_still_binds_all_events():
    q = np.zeros((300, 1, 43))
    q[:, :, 2] = 0.793
    q[:, :, 3] = 1.0
    commands = np.zeros((300, 1, 3))
    commands[:, :, 0] = 2.0
    commands[60, 0, 0] = 0.8
    commands[80, 0, 0] = 0.0
    backend = object.__new__(_StreamingBackend)
    backend.navigation = config()
    backend._planner = HeightPlanner()
    backend.planner_calls = 0
    backend.events = []
    backend.facing = 0.0
    backend.command = tuple(commands[0, 0])
    backend.reference = backend._generate_reference(q[0, 0, :36])
    for frame in range(20, 300, 20):
        backend.command = tuple(commands[frame, 0])
        backend.refresh_reference(None, frame)
    trace = dict(canonical_qpos=q, navigation_command=commands)
    session = HeightPlanner()
    assert (
        len(
            replay_locomotion_segments(
                session,
                trace,
                backend.events,
                3,
                reference_height_probe_m=0.75,
                fast_intercept_probe=True,
            )
        )
        == 15
    )
    assert session.heights[3:5] == [-1.0, -1.0]
    for kw in (
        dict(reference_height_probe_m=0.75),
        dict(reference_height_probe_m=0.7, fast_intercept_probe=True),
        dict(reference_height_probe_m=0.75, fast_intercept_probe=1),
    ):
        session = HeightPlanner()
        with pytest.raises(ValueError):
            replay_locomotion_segments(session, trace, backend.events, 3, **kw)
        assert session.calls == 0


@pytest.mark.parametrize("probe,mode,height", [(False, 3, 0.75), (True, 2, 0.75), (True, 3, 0.7)])
def test_native_fast_profile_rejects_mixed_modes(probe, mode, height):
    with pytest.raises(ValueError, match="fast interception"):
        run_native_first_touch_episode(
            argparse.Namespace(),
            course=(2.5, 0.0, 0.6),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            control_profile=FastInterceptControl(),
            foundation_approach_probe=probe,
            locomotion_mode_probe=mode,
            reference_height_probe_m=height,
        )


def test_valid_fast_profile_reaches_allocator(tmp_path, monkeypatch):
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
            course=(2.5, 0.0, 0.6),
            partition="DECLARED_DEVELOPMENT_DIAGNOSTIC",
            entry_source_path=Path(__file__),
            consumed_bank_hash=None,
            control_profile=FastInterceptControl(),
            foundation_approach_probe=True,
            locomotion_mode_probe=3,
            reference_height_probe_m=0.75,
            approach_mode="bounded_approach_diagnostic",
        )
