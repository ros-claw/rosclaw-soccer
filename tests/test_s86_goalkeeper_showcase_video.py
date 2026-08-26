from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw_soccer.media.goalkeeper_showcase_video import (
    validate_collision_faithful_goalkeeper_manifest,
    validate_goalkeeper_showcase_manifest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _manifest(tmp_path: Path) -> Path:
    video = tmp_path / "showcase.mp4"
    source = tmp_path / "source.json"
    video.write_bytes(b"video")
    source.write_text("{}", encoding="utf-8")
    payload = {
        "schema_version": "rosclaw_soccer.goalkeeper_showcase_video.v1",
        "video_path": str(video),
        "video_hash": hash_bytes(video.read_bytes()),
        "source_files": {str(source): hash_bytes(source.read_bytes())},
        "strict_primary_save_count": 4,
        "controlled_dive_source_status": "REJECTED_NO_SAFE_CANDIDATE",
        "controlled_dive_clip_claim": (
            "INDIVIDUAL_STABLE_QUALIFIED_SAVE_NOT_POLICY_PROMOTION"
        ),
        "controlled_dive_rollout": {
            "first_save": True,
            "first_hand_save": True,
            "qualified_save": True,
            "recovered": True,
            "stable_save": True,
        },
        "double_save_source_status": "REJECTED_BY_CPU_MUJOCO_EXAM",
        "double_save_clip_claim": "INDIVIDUAL_PASSED_ROLLOUT_NOT_POLICY_PROMOTION",
        "double_save_rollout": {
            "first_save": True,
            "first_hand_save": False,
            "recovered": True,
            "second_save": True,
            "second_hand_save": True,
        },
        "fall_then_second_save_included": False,
        "fps": 30,
        "width": 1920,
        "height": 1080,
        "frame_count": 990,
        "duration_sec": 33.0,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    payload["manifest_hash"] = hash_json(payload)
    path = tmp_path / "showcase.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_goalkeeper_showcase_manifest_is_content_bound(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert validate_goalkeeper_showcase_manifest(manifest)["strict_primary_save_count"] == 4
    (tmp_path / "source.json").write_text('{"drifted": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="source binding changed"):
        validate_goalkeeper_showcase_manifest(manifest)


def test_goalkeeper_showcase_rejects_fabricated_fall_second_save(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("manifest_hash")
    payload["fall_then_second_save_included"] = True
    payload["manifest_hash"] = hash_json(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authority contract"):
        validate_goalkeeper_showcase_manifest(manifest)


def test_collision_faithful_showcase_rejects_positive_glove_gap(tmp_path: Path) -> None:
    video = tmp_path / "true-contact.mp4"
    source = tmp_path / "evidence.json"
    video.write_bytes(b"video")
    source.write_text("{}", encoding="utf-8")
    contacts = [
        {
            "lane_id": f"lane-{index}",
            "time_sec": 8.0,
            "position_m": [6.7, 0.1 * index, 1.4],
            "signed_surface_distance_m": -0.001,
            "glove_side": "left" if index == 0 else "right",
        }
        for index in range(4)
    ]
    payload = {
        "schema_version": "rosclaw_soccer.collision_faithful_goalkeeper_video.v1",
        "video_path": str(video),
        "video_hash": hash_bytes(video.read_bytes()),
        "source_files": {str(source): hash_bytes(source.read_bytes())},
        "exact_glove_contacts": contacts,
        "strict_save_count": 4,
        "exact_contact_trace_hz": 500,
        "positive_surface_separation_gate_m": 0.001,
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    payload["manifest_hash"] = hash_json(payload)
    manifest = tmp_path / "true-contact.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_collision_faithful_goalkeeper_manifest(manifest)["strict_save_count"] == 4

    payload.pop("manifest_hash")
    payload["exact_glove_contacts"][0]["signed_surface_distance_m"] = 0.01
    payload["manifest_hash"] = hash_json(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authority contract"):
        validate_collision_faithful_goalkeeper_manifest(manifest)
