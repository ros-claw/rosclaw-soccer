from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw_soccer.media.dynamic_aerial_lunge_video import (
    render_dynamic_aerial_lunge_video,
    validate_dynamic_aerial_lunge_video_manifest,
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
        "schema_version": "rosclaw_soccer.dynamic_aerial_lunge_video.v1",
        "video_path": str(video),
        "video_hash": hash_bytes(video.read_bytes()),
        "source_files": {
            str(evidence): hash_bytes(evidence.read_bytes()),
            str(trajectory): hash_bytes(trajectory.read_bytes()),
        },
        "claim": "STABLE_LATE_LUNGE_TIP_OVER_NOT_AIRBORNE_DIVE",
        "strict_replay": True,
        "true_glove_contact": True,
        "post_save_recovered": True,
        "airborne_claimed": False,
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


def test_dynamic_lunge_video_manifest_is_content_bound(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert validate_dynamic_aerial_lunge_video_manifest(manifest)["strict_replay"] is True
    (tmp_path / "reel.mp4").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="video hash"):
        validate_dynamic_aerial_lunge_video_manifest(manifest)


def test_dynamic_lunge_video_rejects_output_inside_checkout(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    checkout = tmp_path / "checkout"
    with pytest.raises(ValueError, match="output contract"):
        render_dynamic_aerial_lunge_video(
            evidence_path=evidence,
            asset_root=tmp_path,
            output_path=checkout / "reel.mp4",
            source_checkout=checkout,
        )


def test_dynamic_lunge_video_rejects_unqualified_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"passed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="not render eligible"):
        render_dynamic_aerial_lunge_video(
            evidence_path=evidence,
            asset_root=tmp_path,
            output_path=tmp_path / "reel.mp4",
            source_checkout=tmp_path / "checkout",
        )
