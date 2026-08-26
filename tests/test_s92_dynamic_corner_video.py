from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw_soccer.media.dynamic_corner_video import (
    render_dynamic_corner_video,
    validate_dynamic_corner_video_manifest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _manifest(tmp_path: Path) -> Path:
    video = tmp_path / "corner.mp4"
    evidence = tmp_path / "evidence.json"
    video.write_bytes(b"video")
    evidence.write_bytes(b"evidence")
    payload = {
        "schema_version": "rosclaw_soccer.dynamic_corner_video.v1",
        "video_path": str(video),
        "video_hash": hash_bytes(video.read_bytes()),
        "source_files": {str(evidence): hash_bytes(evidence.read_bytes())},
        "claim": "STRICT_MULTI_CORNER_AIRBORNE_SAVE_PORTFOLIO",
        "case_count": 4,
        "strict_replay": True,
        "fps": 60,
        "width": 1920,
        "height": 1080,
        "frame_count": 1200,
        "duration_sec": 20.0,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
    }
    payload["manifest_hash"] = hash_json(payload)
    path = tmp_path / "corner.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_dynamic_corner_video_manifest_is_content_bound(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert validate_dynamic_corner_video_manifest(manifest)["case_count"] == 4
    (tmp_path / "evidence.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source binding"):
        validate_dynamic_corner_video_manifest(manifest)


def test_dynamic_corner_video_rejects_output_inside_checkout(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    checkout = tmp_path / "checkout"
    with pytest.raises(ValueError, match="output contract"):
        render_dynamic_corner_video(
            evidence_path=evidence,
            asset_root=tmp_path,
            output_path=checkout / "corner.mp4",
            source_checkout=checkout,
        )
