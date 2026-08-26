from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw_soccer.evidence.opentrack_exam import (
    OpenTrackEpisodeSpec,
    OpenTrackFoundationExamPlan,
    sanitize_opentrack_adapter_export_config,
    summarize_opentrack_episodes,
)
from rosclaw_soccer.skills.athlete_foundation.foundation_shootout import (
    FoundationThresholds,
)


def _digest(label: str) -> str:
    return "sha256:" + (label.encode().hex() + "0" * 64)[:64]


def _episode(index: int, *, suite: str) -> OpenTrackEpisodeSpec:
    return OpenTrackEpisodeSpec(
        episode_id=f"episode-{index}",
        suite_id=suite,
        dataset_id="dataset",
        motion_id=f"motion-{index}",
        source_hash=_digest(str(index)),
        license_id="research-only",
        critical=suite == "retention",
    )


def _raw(suite: str) -> dict[str, object]:
    return {
        "suite_id": suite,
        "success": True,
        "fell": False,
        "finite_state": True,
        "joint_squared_error_sum": 0.29,
        "joint_error_count": 29,
        "keypoint_squared_error_sum": 0.01,
        "keypoint_error_count": 10,
        "foot_slip_sum_mps": 0.02,
        "foot_slip_count": 2,
        "minimum_pelvis_height_m": 0.72,
        "peak_torque_fraction": 0.8,
        "saturated_control_steps": 0,
        "control_steps": 100,
        "root_angular_speeds_rad_s": [0.2, 0.3],
        "joint_jerk_squared_sum": 29.0,
        "joint_jerk_count": 29,
        "transition_error_rad": 0.1,
        "recovered_upright": True,
    }


def test_plan_requires_matched_retention_and_acquisition_suites() -> None:
    episodes = tuple(_episode(index, suite="retention") for index in range(8))

    with pytest.raises(ValueError, match="both retention and acquisition"):
        OpenTrackFoundationExamPlan(episodes=episodes)


def test_plan_is_sim_only_and_content_addressed() -> None:
    episodes = tuple(
        _episode(index, suite="retention" if index < 4 else "acquisition") for index in range(8)
    )
    plan = OpenTrackFoundationExamPlan(episodes=episodes)

    assert plan.plan_hash.startswith("sha256:")
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(plan, hardware_command_sent=True)


def test_summary_passes_complete_finite_matched_physics() -> None:
    reports = tuple(_raw("retention" if index < 4 else "acquisition") for index in range(8))

    metrics, reasons, suites = summarize_opentrack_episodes(
        reports, thresholds=FoundationThresholds()
    )

    assert metrics.tracking_success_rate == 1.0
    assert reasons == ()
    assert suites["retention"]["episode_count"] == 4
    assert suites["acquisition"]["success_rate"] == 1.0


def test_summary_fails_closed_on_fall_and_missing_physics() -> None:
    reports = [_raw("retention" if index < 4 else "acquisition") for index in range(8)]
    reports[0]["success"] = False
    reports[0]["minimum_pelvis_height_m"] = 0.30
    metrics, reasons, _ = summarize_opentrack_episodes(
        tuple(reports), thresholds=FoundationThresholds()
    )

    assert metrics.tracking_success_rate == 0.875
    assert "tracking_success_below_floor" in reasons
    assert "pelvis_height_below_floor" in reasons
    with pytest.raises(ValueError, match="at least eight"):
        summarize_opentrack_episodes(tuple(reports[:7]), thresholds=FoundationThresholds())


def test_export_config_sanitizer_removes_only_callbacks(tmp_path: Path) -> None:
    source = tmp_path / "config.json"
    target = tmp_path / "export" / "config.json"
    source.write_text(
        json.dumps(
            {
                "env_config": {"history_len": 79},
                "policy_config": {
                    "num_timesteps": 5_000_000,
                    "network_factory": {"policy_hidden_layer_sizes": [512, 256]},
                    "progress_fn": "function callback",
                    "randomization_fn": "function domain_randomize",
                    "wrap_env_fn": None,
                },
                "mbppo_policy_config": {"use_adapter": True},
            }
        ),
        encoding="utf-8",
    )

    evidence = sanitize_opentrack_adapter_export_config(source_path=source, output_path=target)

    sanitized = json.loads(target.read_text(encoding="utf-8"))
    assert sanitized["policy_config"] == {
        "num_timesteps": 5_000_000,
        "network_factory": {"policy_hidden_layer_sizes": [512, 256]},
    }
    assert evidence["removed_fields"] == [
        "progress_fn",
        "randomization_fn",
        "wrap_env_fn",
    ]
    assert evidence["network_fields_changed"] is False
    assert evidence["source_hash"] != evidence["output_hash"]


def test_export_config_sanitizer_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "config.json"
    source.write_text('{"policy_config":{"progress_fn":"callback"}}', encoding="utf-8")

    with pytest.raises(ValueError, match="new output"):
        sanitize_opentrack_adapter_export_config(source_path=source, output_path=source)
