from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw_soccer.media.aerial_curriculum_video import (
    validate_aerial_curriculum_video_manifest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.goalkeeper_aerial_curriculum import (
    GoalkeeperAerialCurriculumCase,
    GoalkeeperAerialCurriculumConfig,
    validate_goalkeeper_aerial_curriculum,
)

_EVIDENCE = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s80-low-to-true-high-aerial-curriculum-v6/goalkeeper-aerial-curriculum-exam.json"
)
_VIDEO_MANIFEST = _EVIDENCE.with_name("g1-low-to-true-high-aerial-corner-growth.json")


def test_aerial_curriculum_orders_low_to_high_and_keeps_frontier_out_of_gate() -> None:
    config = GoalkeeperAerialCurriculumConfig()
    required = tuple(case for case in config.aerial_cases if case.required)
    frontier = tuple(case for case in config.aerial_cases if not case.required)

    assert {case.band for case in required} == {
        "MID",
        "HIGH_CENTER",
        "HIGH_INNER",
        "HIGH_CORNER",
    }
    assert len(required) == 9
    assert len(frontier) == 2
    assert all(case.band == "HIGH_CORNER" and case.flight_sec == 1.40 for case in frontier)
    true_high = tuple(case for case in required if case.minimum_contact_height_m > 0.0)
    assert len(true_high) == 2
    assert all(
        case.launch_vertical_bias_mps == 0.30 and case.minimum_contact_height_m == 1.30
        for case in true_high
    )
    assert config.activation_ceiling == "SIM_ONLY"
    assert not config.hardware_authorized
    assert config.actor_threat_warmup_sec == 0.04
    assert config.actor_minimum_intercept_confidence == 0.25
    assert config.actor_operational_space_reach_local_x_m == 0.0
    assert config.actor_operational_space_reach_side_offset_m == 0.10


def test_aerial_curriculum_rejects_duplicate_or_unbounded_cases() -> None:
    case = GoalkeeperAerialCurriculumCase("bounded", "MID", 0.7, 1.2, 1.2)
    with pytest.raises(ValueError, match="config is invalid"):
        GoalkeeperAerialCurriculumConfig(aerial_cases=(case,) * 7)
    with pytest.raises(ValueError, match="case is invalid"):
        replace(case, target_z_m=1.80)
    with pytest.raises(ValueError, match="case is invalid"):
        replace(case, minimum_contact_height_m=1.30)


def test_aerial_curriculum_router_is_bounded_and_content_hashed() -> None:
    config = GoalkeeperAerialCurriculumConfig()

    assert config.config_hash.startswith("sha256:")
    with pytest.raises(ValueError, match="config is invalid"):
        replace(config, actor_threat_warmup_sec=0.01)
    with pytest.raises(ValueError, match="config is invalid"):
        replace(config, actor_minimum_intercept_confidence=0.20)
    with pytest.raises(ValueError, match="config is invalid"):
        replace(config, actor_operational_space_reach_local_x_m=0.40)
    with pytest.raises(ValueError, match="config is invalid"):
        replace(
            config,
            actor_operational_space_reach_local_x_m=0.30,
            actor_operational_space_reach_side_offset_m=0.10,
        )
    with pytest.raises(ValueError, match="config is invalid"):
        replace(config, activation_ceiling="HARDWARE")


def test_aerial_video_manifest_is_downstream_and_tamper_evident(tmp_path: Path) -> None:
    video = tmp_path / "growth.mp4"
    report = tmp_path / "exam.json"
    video.write_bytes(b"scored-trajectory-pixels")
    report.write_text("{}\n", encoding="utf-8")
    payload = {
        "schema_version": "rosclaw_soccer.aerial_curriculum_video.v1",
        "video_path": str(video),
        "video_hash": "sha256:" + hashlib.sha256(video.read_bytes()).hexdigest(),
        "report_path": str(report),
        "report_file_hash": hash_bytes(report.read_bytes()),
        "fps": 30,
        "width": 1920,
        "height": 1080,
        "frame_count": 60,
        "duration_sec": 2.0,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    payload["manifest_hash"] = hash_json(payload)
    manifest = tmp_path / "growth.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_aerial_curriculum_video_manifest(manifest)["frame_count"] == 60
    video.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="authority contract"):
        validate_aerial_curriculum_video_manifest(manifest)


@pytest.mark.skipif(not _EVIDENCE.is_file(), reason="S80 aerial evidence unavailable")
def test_actual_aerial_curriculum_is_current_and_physically_height_gated() -> None:
    report = validate_goalkeeper_aerial_curriculum(_EVIDENCE)

    assert report["strict_replay"] is True
    assert report["metrics"]["low_pass_rate"] == 1.0
    assert report["metrics"]["true_high_corner_pass_rate"] == 1.0
    assert report["metrics"]["minimum_true_high_contact_height_m"] >= 1.30
    assert report["failure_memory"] == []
    assert report["promotion_eligible"] is False


@pytest.mark.skipif(not _VIDEO_MANIFEST.is_file(), reason="S80 aerial video unavailable")
def test_actual_aerial_video_is_bound_and_visualization_only() -> None:
    manifest = validate_aerial_curriculum_video_manifest(_VIDEO_MANIFEST)

    assert manifest["width"] == 1920
    assert manifest["height"] == 1080
    assert manifest["pixels_used_for_scoring"] is False
    assert manifest["promotion_eligible"] is False
