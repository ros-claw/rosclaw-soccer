from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw_soccer.media.alternating_growth_video import (
    render_alternating_growth_video,
    validate_alternating_growth_video_manifest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _manifest(tmp_path: Path) -> Path:
    video = tmp_path / "growth.mp4"
    pass_evidence = tmp_path / "pass.json"
    strike_evidence = tmp_path / "strike.json"
    video.write_bytes(b"video")
    pass_evidence.write_bytes(b"pass")
    strike_evidence.write_bytes(b"strike")
    payload = {
        "schema_version": "rosclaw_soccer.alternating_growth_video.v1",
        "video_path": str(video),
        "video_hash": hash_bytes(video.read_bytes()),
        "source_files": {
            str(pass_evidence): hash_bytes(pass_evidence.read_bytes()),
            str(strike_evidence): hash_bytes(strike_evidence.read_bytes()),
        },
        "claim": "ALTERNATING_TEAM_GROWTH_PASSER_THEN_SHOOTER",
        "case_count": 4,
        "source_evidence_passed": True,
        "fps": 30,
        "width": 1920,
        "height": 1080,
        "frame_count": 600,
        "duration_sec": 20.0,
        "clips": [],
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
    }
    payload["manifest_hash"] = hash_json(payload)
    path = tmp_path / "growth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_alternating_growth_manifest_is_content_bound(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert validate_alternating_growth_video_manifest(manifest)["case_count"] == 4
    (tmp_path / "strike.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source binding"):
        validate_alternating_growth_video_manifest(manifest)


def test_alternating_growth_video_rejects_output_inside_checkout(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    checkout = tmp_path / "checkout"
    with pytest.raises(ValueError, match="output contract"):
        render_alternating_growth_video(
            dynamic_pass_evidence_path=evidence,
            upper_corner_evidence_path=evidence,
            asset_root=tmp_path,
            output_path=checkout / "growth.mp4",
            source_checkout=checkout,
        )
