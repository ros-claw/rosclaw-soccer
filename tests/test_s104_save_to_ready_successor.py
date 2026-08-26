from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from rosclaw_soccer.media.save_to_ready_video import (
    validate_save_to_ready_video_manifest,
)
from rosclaw_soccer.skills.team.shared_world import (
    G1GoalkeeperConfig,
    G1SharedWorldResult,
)
from rosclaw_soccer.training.save_to_ready_successor import (
    SaveToReadySuccessorConfig,
    evaluate_save_to_ready_successor,
    validate_save_to_ready_successor_evidence,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec

_EVIDENCE = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s104-save-to-ready-successor-v3/evidence.json"
)
_VIDEO_MANIFEST = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s104-save-to-ready-showcase-v4/s104-save-absorb-reengage-ready.json"
)


def _successful_result() -> G1SharedWorldResult:
    return cast(
        G1SharedWorldResult,
        SimpleNamespace(
            goalkeeper_ball_contact_time_sec=8.0,
            goalkeeper_save_observed=True,
            finite_state=True,
            goalkeeper_joint_limit_violation=False,
        ),
    )


def _trajectory() -> dict[str, np.ndarray]:
    time = np.arange(0.02, 25.001, 0.02, dtype=np.float64)
    count = time.size
    pelvis = np.zeros((count, 7), dtype=np.float64)
    pelvis[:, 0] = 7.02
    pelvis[:, 2] = 0.77
    pelvis[:, 3] = 1.0
    root_velocity = np.zeros((count, 6), dtype=np.float64)
    torso = np.zeros((count, 4), dtype=np.float64)
    torso[:, 3] = 1.0
    support = np.ones((count, 2), dtype=np.bool_)
    command = np.zeros(count, dtype=np.float64)
    probe = (time >= 18.0) & (time < 18.8)
    probe_indices = np.flatnonzero(probe)
    probe_velocity = np.minimum(0.20, np.arange(probe_indices.size) * 0.02)
    root_velocity[probe_indices, 1] = probe_velocity
    command[probe] = 0.14
    for previous, current in zip(probe_indices[:-1], probe_indices[1:], strict=True):
        pelvis[current:, 1] += root_velocity[previous, 1] * (time[current] - time[previous])
    return {
        "time": time,
        "goalkeeper_pelvis_pose": pelvis,
        "goalkeeper_root_velocity": root_velocity,
        "goalkeeper_torso_quaternion": torso,
        "goalkeeper_foot_contact": support,
        "goalkeeper_joint_position": np.zeros((count, 29), dtype=np.float64),
        "goalkeeper_policy_action": np.zeros((count, 29), dtype=np.float64),
        "goalkeeper_command_mps": command,
    }


def test_successor_config_is_strictly_sim_only() -> None:
    config = SaveToReadySuccessorConfig()
    assert config.activation_ceiling == "SIM_ONLY"
    assert not config.hardware_authorized
    assert config.maximum_root_linear_speed_mps == 0.25
    assert config.maximum_root_angular_speed_rad_s == 0.50
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="thresholds"):
        replace(config, maximum_root_linear_speed_mps=0.26)


def test_goalkeeper_probe_requires_explicit_bounded_recovery() -> None:
    with pytest.raises(ValueError, match="requires contact stabilization"):
        G1GoalkeeperConfig(post_contact_ready_recovery_enabled=True)
    with pytest.raises(ValueError, match="successor probe"):
        G1GoalkeeperConfig(
            successor_lateral_probe_enabled=True,
            successor_lateral_probe_command_mps=0.14,
        )
    with pytest.raises(ValueError, match="successor probe"):
        G1GoalkeeperConfig(successor_lateral_probe_command_mps=0.04)
    valid = G1GoalkeeperConfig(
        post_contact_stabilization_enabled=True,
        post_contact_ready_recovery_enabled=True,
        post_contact_ready_lateral_deadband_m=0.15,
        successor_lateral_probe_enabled=True,
        successor_lateral_probe_command_mps=0.14,
    )
    assert valid.successor_lateral_probe_enabled


def test_successor_evaluator_requires_ready_probe_and_ready_again() -> None:
    trajectory = _trajectory()
    report = evaluate_save_to_ready_successor(
        result=_successful_result(),
        trajectory=trajectory,
        goal=G1TrainingGoalSpec(plane_x_m=7.5, width_m=7.32, height_m=2.44),
        depth_from_goal_line_m=0.48,
        expected_probe_command_mps=0.14,
    )
    assert report["passed"] is True
    assert report["pre_probe_ready"]["passed"] is True
    assert report["probe"]["gates"]["lateral_acceleration_capacity"] is True
    assert report["post_probe_ready"]["passed"] is True
    assert report["successor_state"] == "GOALKEEPER_READY"
    assert report["reset_or_teleport_used"] is False

    trajectory["goalkeeper_root_velocity"][-50:, 3] = 0.51
    rejected = evaluate_save_to_ready_successor(
        result=_successful_result(),
        trajectory=trajectory,
        goal=G1TrainingGoalSpec(plane_x_m=7.5, width_m=7.32, height_m=2.44),
        depth_from_goal_line_m=0.48,
        expected_probe_command_mps=0.14,
    )
    assert rejected["passed"] is False
    assert rejected["post_probe_ready"]["gates"]["low_root_angular_speed"] is False


def test_successor_evaluator_rejects_nonfinite_telemetry() -> None:
    trajectory = _trajectory()
    trajectory["goalkeeper_pelvis_pose"][100, 2] = np.nan
    report = evaluate_save_to_ready_successor(
        result=_successful_result(),
        trajectory=trajectory,
        goal=G1TrainingGoalSpec(plane_x_m=7.5, width_m=7.32, height_m=2.44),
        depth_from_goal_line_m=0.48,
        expected_probe_command_mps=0.14,
    )
    assert report == {"passed": False, "reason": "successor telemetry is invalid"}


def test_current_s104_evidence_is_content_bound_when_available() -> None:
    if not _EVIDENCE.is_file():
        pytest.skip("S104 external CPU MuJoCo evidence is unavailable")
    report = validate_save_to_ready_successor_evidence(_EVIDENCE)
    assert report["passed"] is True
    assert report["portfolio_gates"]["all_post_probe_goalkeeper_ready"] is True
    assert all(
        case["successor"]["successor_state"] == "GOALKEEPER_READY"
        for case in report["cases"].values()
    )


def test_current_s104_video_is_content_bound_when_available() -> None:
    if not _VIDEO_MANIFEST.is_file():
        pytest.skip("S104 external evidence video is unavailable")
    manifest = validate_save_to_ready_video_manifest(_VIDEO_MANIFEST)
    assert manifest["strict_replay"] is True
    assert manifest["all_post_probe_goalkeeper_ready"] is True
    assert manifest["pixels_used_for_scoring"] is False
