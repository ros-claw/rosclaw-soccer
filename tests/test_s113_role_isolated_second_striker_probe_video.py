from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw_soccer.media.role_isolated_second_striker_probe_video import (
    validate_role_isolated_second_striker_probe_video,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_MANIFEST = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s113-role-isolated-target-actor-control-video-v2/"
    "s113-safe-rejection-review.json"
)


def test_role_isolated_video_validator_fails_closed_on_authority(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    video = tmp_path / "review.mp4"
    video.write_bytes(b"video")
    manifest = tmp_path / "review.json"
    payload = {
        "schema_version": "rosclaw_soccer.role_isolated_second_striker_probe_video.v1",
        "video_path": str(video),
        "video_hash": hash_bytes(video.read_bytes()),
        "source_files": {str(source): hash_bytes(source.read_bytes())},
        "claim": "ROLE_ISOLATED_SECOND_STRIKER_REJECTED_CANDIDATE_REVIEW_VIDEO",
        "evidence_passed": True,
        "candidate_promoted": False,
        "candidate_status": "REJECTED_NO_SUPPORTED_PLASTICITY",
        "strict_replay": True,
        "complete_chain_retained": True,
        "candidate_selected_frame_count": 0,
        "four_g1_visible": True,
        "two_physical_balls_visible": True,
        "two_physical_saves": True,
        "fps": 30,
        "width": 1920,
        "height": 1080,
        "frame_count": 300,
        "duration_sec": 10.0,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": True,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "implementation_hash": "sha256:not-current",
    }
    payload["manifest_hash"] = hash_json(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="bindings"):
        validate_role_isolated_second_striker_probe_video(manifest)


def test_external_s113_role_isolated_video_if_available() -> None:
    if not _MANIFEST.is_file():
        pytest.skip("external S113 role-isolated stage video is not present")
    payload = validate_role_isolated_second_striker_probe_video(_MANIFEST)
    assert payload["candidate_promoted"] is False
    assert payload["candidate_selected_frame_count"] == 0
    assert payload["pixels_used_for_scoring"] is False
