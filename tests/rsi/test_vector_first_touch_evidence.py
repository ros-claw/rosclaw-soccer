"""Vector contact evidence must count physical episodes, not control frames."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.vector_first_touch_evidence import (
    audit_reset_replay,
    audit_vector_first_touch,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _fixture(folder: Path) -> None:
    folder.mkdir()
    positions = np.zeros((50, 2, 3))
    positions[:, 0] = (2.4, -0.1, 0.11)
    positions[:, 1] = (2.6, 7.9, 0.11)
    force = np.zeros((50, 2, 6))
    force[40, 0, 0] = 10.0
    trace = folder / "trace.npz"
    np.savez_compressed(
        trace,
        ball_position_m=positions,
        ball_angular_velocity_rad_s=np.zeros((50, 2, 3)),
        ball_body_contact_force_peak_n=force,
    )
    report = {
        "schema": "rsi_isaac_vector_first_touch_smoke_v1",
        "activation_ceiling": "SIM_ONLY",
        "learning_authorized": False,
        "promotion_authorized": False,
        "frames": 50,
        "source_hash": hash_json("runner"),
        "trace_hash": hash_bytes(trace.read_bytes()),
        "environments": [
            {
                "environment": 0,
                "lane_y_m": 0.0,
                "course": {"ball_x_m": 2.4, "ball_y_local_m": -0.1, "ball_vx_m_s": -0.5},
                "minimum_pelvis_z_m": 0.7,
                "first_contact_frame": 40,
                "contact_body_indices": [0],
                "ball_final_local_xyz_m": [2.4, -0.1, 0.11],
            },
            {
                "environment": 1,
                "lane_y_m": 8.0,
                "course": {"ball_x_m": 2.6, "ball_y_local_m": -0.1, "ball_vx_m_s": 0.5},
                "minimum_pelvis_z_m": 0.7,
                "first_contact_frame": None,
                "contact_body_indices": [],
                "ball_final_local_xyz_m": [2.6, -0.1, 0.11],
            },
        ],
    }
    report["report_hash"] = hash_json(report)
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_vector_audit_counts_independent_episodes(tmp_path: Path) -> None:
    _fixture(tmp_path / "case")
    result = audit_vector_first_touch(tmp_path / "case")
    assert result["independent_physical_episode_count"] == 2
    assert result["any_contact_episode_count"] == 1
    assert result["clean_foot_only_episode_count"] == 1
    assert result["positive_foot_sides"] == ["left"]
    assert result["imitation_training_authorized"] is False


def test_vector_audit_rejects_tamper_and_duplicate(tmp_path: Path) -> None:
    folder = tmp_path / "case"
    _fixture(folder)
    with (folder / "trace.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="unauthenticated"):
        audit_vector_first_touch(folder)
    _fixture(tmp_path / "other")
    folder = tmp_path / "other"
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    report["environments"][1]["course"] = report["environments"][0]["course"]
    report.pop("report_hash")
    report["report_hash"] = hash_json(report)
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate courses"):
        audit_vector_first_touch(folder)


def test_in_process_reset_replay_is_not_new_course(tmp_path: Path) -> None:
    folder = tmp_path / "case"
    _fixture(folder)
    source = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    with np.load(folder / "trace.npz", allow_pickle=False) as original:
        np.savez_compressed(folder / "reset_replay.npz", **{k: original[k] for k in original.files})
    report = {
        "schema": "rsi_isaac_vector_first_touch_reset_replay_v1",
        "activation_ceiling": "SIM_ONLY",
        "learning_authorized": False,
        "promotion_authorized": False,
        "source_report_hash": source["report_hash"],
        "source_hash": source.get("source_hash"),
        "replay_trace_hash": hash_bytes((folder / "reset_replay.npz").read_bytes()),
        "reset_verified": True,
        "first_contact_frames": [40, None],
        "replay_first_contact_frames": [40, None],
        "body_classes_equal": True,
        "precontact_max_ball_position_diff_m": 0.0,
        "full_ball_position_max_diff_m": 0.0,
        "minimum_replay_pelvis_z_m": 0.7,
    }
    report["report_hash"] = hash_json(report)
    (folder / "reset_report.json").write_text(json.dumps(report), encoding="utf-8")
    result = audit_reset_replay(folder)
    assert result["reset_verified"] is True
    assert result["independent_new_course_count"] == 0
    with (folder / "reset_replay.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="unauthenticated"):
        audit_reset_replay(folder)
