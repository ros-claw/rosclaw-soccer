from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw_soccer.media.first_touch_growth_video import (
    render_first_touch_growth_video,
    validate_first_touch_growth_video_manifest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _manifest(tmp_path: Path) -> Path:
    video = tmp_path / "first-touch.mp4"
    evidence = tmp_path / "paired-exam.json"
    video.write_bytes(b"video")
    evidence.write_bytes(b"evidence")
    payload = {
        "schema_version": "rosclaw_soccer.first_touch_growth_video.v1",
        "claim": "MATCHED_FIRST_TOUCH_LOCAL_ACQUISITION_BEFORE_AFTER",
        "video_path": str(video),
        "video_hash": hash_bytes(video.read_bytes()),
        "source_files": {str(evidence): hash_bytes(evidence.read_bytes())},
        "source_exam_hash": "sha256:exam",
        "source_exam_passed": True,
        "fps": 30,
        "width": 1920,
        "height": 1080,
        "frame_count": 300,
        "duration_sec": 10.0,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
    }
    payload["manifest_hash"] = hash_json(payload)
    path = tmp_path / "first-touch.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_first_touch_video_manifest_is_content_bound(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert validate_first_touch_growth_video_manifest(manifest)["source_exam_passed"] is True
    (tmp_path / "paired-exam.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source changed"):
        validate_first_touch_growth_video_manifest(manifest)


def test_first_touch_video_rejects_output_inside_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    with pytest.raises(ValueError, match="output contract"):
        render_first_touch_growth_video(
            paired_exam_path=tmp_path / "exam.json",
            baseline_report_path=tmp_path / "baseline.json",
            candidate_report_path=tmp_path / "candidate.json",
            asset_root=tmp_path,
            output_path=checkout / "first-touch.mp4",
            source_checkout=checkout,
        )
