from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.media.dynamic_takeoff_video import (
    _observed_mask_duration_sec,
    _timeline,
    render_dynamic_takeoff_video,
    validate_dynamic_takeoff_video_manifest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _manifest(tmp_path: Path) -> Path:
    video = tmp_path / "reel.mp4"
    evidence = tmp_path / "evidence.json"
    trajectory = tmp_path / "trajectory.npz"
    video.write_bytes(b"video")
    evidence.write_bytes(b"evidence")
    trajectory.write_bytes(b"trajectory")
    payload = {
        "schema_version": "rosclaw_soccer.dynamic_takeoff_video.v1",
        "video_path": str(video),
        "video_hash": hash_bytes(video.read_bytes()),
        "source_files": {
            str(evidence): hash_bytes(evidence.read_bytes()),
            str(trajectory): hash_bytes(trajectory.read_bytes()),
        },
        "claim": "TRUE_AIRBORNE_SAVE_WITH_FOOT_CONTACT_GROUNDED_LANDING",
        "strict_replay": True,
        "true_airborne": True,
        "foot_contact_grounded": True,
        "true_glove_contact": True,
        "bounded_landing": True,
        "post_save_recovered": True,
        "commercial_use_allowed": False,
        "fps": 60,
        "width": 1920,
        "height": 1080,
        "frame_count": 600,
        "duration_sec": 10.0,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    payload["manifest_hash"] = hash_json(payload)
    path = tmp_path / "reel.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_dynamic_takeoff_video_manifest_is_content_bound(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert validate_dynamic_takeoff_video_manifest(manifest)["true_airborne"] is True
    (tmp_path / "trajectory.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source binding"):
        validate_dynamic_takeoff_video_manifest(manifest)


def test_dynamic_takeoff_video_rejects_output_inside_checkout(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    checkout = tmp_path / "checkout"
    with pytest.raises(ValueError, match="output contract"):
        render_dynamic_takeoff_video(
            evidence_path=evidence,
            asset_root=tmp_path,
            output_path=checkout / "reel.mp4",
            source_checkout=checkout,
        )


def test_dynamic_takeoff_video_rejects_unqualified_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"passed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="not render eligible"):
        render_dynamic_takeoff_video(
            evidence_path=evidence,
            asset_root=tmp_path,
            output_path=tmp_path / "reel.mp4",
            source_checkout=tmp_path / "checkout",
        )


def test_dynamic_takeoff_timeline_discloses_landing_capture() -> None:
    time = np.arange(0.0, 10.02, 0.02)
    capture = (time >= 8.20) & (time <= 8.40)
    trajectory = {
        "time": time,
        "goalkeeper_landing_capture_active": capture,
    }
    result = {
        "pass_contact_time_sec": 5.60,
        "shot_contact_time_sec": 7.50,
        "goalkeeper_glove_contact_time_sec": 8.01,
        "goalkeeper_glove_contact_surface_distance_m": -0.00047,
    }
    metrics = {
        "airborne_start_sec": 8.02,
        "airborne_stop_sec": 8.20,
        "landing_time_sec": 8.20,
        "airborne_duration_sec": 0.18,
        "flight_pelvis_rise_m": 0.0456,
        "landing_vertical_speed_mps": 0.916,
    }

    clips = _timeline(result, metrics, trajectory, 60)

    assert _observed_mask_duration_sec(time, capture) == pytest.approx(0.20)
    assert "200 ms PROPRIOCEPTIVE CAPTURE" in clips[4].label
    assert "FEEDBACK FOUNDATION" in clips[5].label
