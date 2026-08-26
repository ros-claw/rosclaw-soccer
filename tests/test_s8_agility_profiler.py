from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.skills.team.agility_profiler import (
    AgilityProfilerConfig,
    AgilityRuntimeTelemetry,
    profile_temporal_agility,
)


def _trajectory() -> dict[str, np.ndarray]:
    time = np.arange(0.0, 0.22, 0.02)
    frames = len(time)
    action = np.zeros((frames, 2), dtype=np.float64)
    action[4:, 0] = np.linspace(0.0, 0.30, frames - 4)
    velocity = np.gradient(action, time, axis=0)
    commanded = np.full((frames, 2), 10.0, dtype=np.float64)
    projected = commanded.copy()
    projected[5:7, 0] = 8.0
    executed = projected.copy()
    executed[6, 1] = 6.0
    return {
        "time": time,
        "goalkeeper_joint_position": action * 0.5,
        "goalkeeper_joint_velocity": velocity,
        "goalkeeper_policy_action": action,
        "goalkeeper_target_velocity": velocity,
        "goalkeeper_commanded_torque": commanded,
        "goalkeeper_safety_projected_torque": projected,
        "goalkeeper_executed_torque": executed,
    }


def test_profiler_attributes_latency_projection_and_execution() -> None:
    profile = profile_temporal_agility(
        _trajectory(),
        role="goalkeeper",
        observation_event_sec=0.04,
        telemetry=AgilityRuntimeTelemetry(
            inference_latency_sec=(0.001, 0.002, 0.003),
            handoff_requested_sec=0.04,
        ),
    )

    assert profile.complete
    assert profile.reaction_latency_sec == pytest.approx(0.06)
    assert profile.skill_handoff_latency_sec == pytest.approx(0.06)
    assert profile.torque_projection_fraction > 0.0
    assert profile.actuator_tracking_miss_fraction > 0.0
    assert "safety_projection_limited" in profile.bottlenecks
    assert "actuator_tracking_limited" in profile.bottlenecks
    assert profile.inference_latency_p90_ms == pytest.approx(2.8)


def test_profiler_marks_missing_runtime_channel_without_inventing_zero_latency() -> None:
    trajectory = _trajectory()
    del trajectory["goalkeeper_executed_torque"]

    profile = profile_temporal_agility(trajectory, role="goalkeeper")

    assert not profile.complete
    assert "executed" in profile.missing_channels
    assert "inference_latency" in profile.missing_channels
    assert profile.inference_latency_p50_ms is None


def test_profiler_rejects_non_monotonic_time() -> None:
    trajectory = _trajectory()
    trajectory["time"][4] = trajectory["time"][3]

    with pytest.raises(ValueError, match="strictly increasing"):
        profile_temporal_agility(trajectory, role="goalkeeper")


def test_velocity_clip_fraction_uses_explicit_joint_limits() -> None:
    profile = profile_temporal_agility(
        _trajectory(),
        role="goalkeeper",
        config=AgilityProfilerConfig(velocity_limits_rad_s=(1.0, 1.0)),
    )

    assert profile.target_velocity_clip_fraction is not None
    assert profile.target_velocity_clip_fraction > 0.0
