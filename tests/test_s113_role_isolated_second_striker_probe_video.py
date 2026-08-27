from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.media.role_isolated_second_striker_probe_video import (
    _timeline,
    validate_role_isolated_second_striker_probe_video,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_MANIFEST = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s116-proprioceptive-recovery-video-v2/"
    "s116-impact-recovery.json"
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


def test_current_role_isolated_video_if_available() -> None:
    if not _MANIFEST.is_file():
        pytest.skip("current role-isolated stage video is not present")
    payload = validate_role_isolated_second_striker_probe_video(_MANIFEST)
    assert payload["candidate_promoted"] is True
    assert payload["candidate_selected_frame_count"] == 1
    assert payload["goalkeeper_proprioceptive_capture_frame_count"] == 61
    assert payload["pixels_used_for_scoring"] is False


def test_s114_timeline_labels_actual_candidate_authority() -> None:
    clips = _timeline(
        {"time": np.asarray((0.0, 23.0))},
        {
            "goalkeeper_glove_contact_time_sec": 8.0,
            "second_threat_rearm_time_sec": 12.0,
            "second_striker_contact_time_sec": 17.0,
            "goalkeeper_second_glove_contact_time_sec": 17.5,
            "second_striker_contact_force_peak_n": 790.0,
            "goalkeeper_glove_contact_height_m": 1.42,
            "goalkeeper_second_glove_contact_height_m": 1.48,
        },
        {
            "frozen_parent_selected_frame_count": 0,
            "candidate_selected_frame_count": 2,
        },
        30,
        promoted=True,
    )

    labels = " ".join(clip.label for clip in clips)
    assert "CANDIDATE SELECTED" in labels
    assert "LEARNED RIGHT-FOOT CONTACT" in labels
    assert "SEALED HOLDOUT STILL REQUIRED" in labels


def test_s115_timeline_discloses_heavy_ball_motion_curriculum() -> None:
    clips = _timeline(
        {"time": np.asarray((0.0, 23.0))},
        {
            "goalkeeper_glove_contact_time_sec": 8.0,
            "second_threat_rearm_time_sec": 12.0,
            "second_striker_contact_time_sec": 17.0,
            "goalkeeper_second_glove_contact_time_sec": 17.5,
            "second_striker_contact_force_peak_n": 832.0,
            "goalkeeper_glove_contact_height_m": 1.42,
            "goalkeeper_second_glove_contact_height_m": 1.37,
        },
        {
            "frozen_parent_selected_frame_count": 0,
            "candidate_selected_frame_count": 1,
        },
        30,
        promoted=True,
        motion_curriculum=True,
    )

    labels = " ".join(clip.label for clip in clips)
    assert "HEAVY-BALL WHOLE-BODY CURRICULUM" in labels
    assert "CURRICULUM BODY PITCH + LEARNED CONTACT ACTOR" in labels
    assert "NEIGHBOR HOLDOUT REQUIRED" in labels


def test_s116_timeline_discloses_proprioceptive_impact_recovery() -> None:
    clips = _timeline(
        {"time": np.asarray((0.0, 23.0))},
        {
            "goalkeeper_glove_contact_time_sec": 8.0,
            "second_threat_rearm_time_sec": 12.0,
            "second_striker_contact_time_sec": 17.0,
            "goalkeeper_second_glove_contact_time_sec": 17.5,
            "second_striker_contact_force_peak_n": 829.0,
            "goalkeeper_glove_contact_height_m": 1.42,
            "goalkeeper_second_glove_contact_height_m": 1.42,
        },
        {
            "frozen_parent_selected_frame_count": 0,
            "candidate_selected_frame_count": 1,
            "goalkeeper_proprioceptive_capture_frame_count": 61,
        },
        30,
        promoted=True,
        motion_curriculum=True,
        proprioceptive_recovery=True,
    )

    labels = " ".join(clip.label for clip in clips)
    assert "PROPRIOCEPTIVE IMPACT RECOVERY" in labels
    assert "61 PHYSICAL CONTROL FRAMES" in labels
    assert "SEALED NEIGHBOR RECOVERED" in labels
