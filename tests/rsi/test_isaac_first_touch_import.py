"""Isaac contact frames must not inflate independent learner episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.isaac_first_touch_import import import_first_touch_curriculum
from rosclaw_soccer.sim.contact_training_samples import BODY_ORDER
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _fixture(folder: Path) -> None:
    folder.mkdir()
    force = np.zeros((3, 10, 6))
    force[2, 0, 0] = 10.0
    trace = folder / "trajectory.npz"
    np.savez_compressed(
        trace,
        ball_observation_position_m=np.zeros((3, 3)),
        ball_observation_velocity_m_s=np.zeros((3, 3)),
        contact_body_position_m=np.zeros((3, 1, 4, 3)),
        contact_body_velocity_m_s=np.zeros((3, 1, 4, 3)),
        ball_body_contact_force_micro_n=force,
    )
    report = {
        "schema": "rosclaw_soccer.rsi.isaac_sonic_stand.v3",
        "trajectory_hash": hash_bytes(trace.read_bytes()),
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "trained_actor": False,
        "ball_rolling_start": True,
        "track_ball_contacts": True,
        "agent_count": 1,
        "forward_command_m_s": 1.4,
        "lateral_command_m_s": 0.0,
        "reactive_lateral_command_m_s": 0.0,
        "right_knee_contact_residual_rad": 0.0,
        "right_hip_pitch_contact_residual_rad": 0.0,
        "right_ankle_pitch_contact_residual_rad": 0.0,
        "ball_body_contact_labels": [
            "left_foot",
            "right_foot",
            "left_ankle_pitch",
            "right_ankle_pitch",
            "left_knee",
            "right_knee",
        ],
        "contact_kinematic_body_names": list(BODY_ORDER),
        "clean_foot_only_contact_verified": True,
        "min_pelvis_height_m": 0.7,
        "individual": {"blue.playmaker": {"joint_projection_count": 0}},
        "agent_ids": ["blue.playmaker"],
        "first_foot_ball_contact_microstep": 20,
        "ball_x_m": 2.4,
        "ball_y_m": -0.1,
        "ball_initial_vx_m_s": -0.5,
        "ball_initial_vy_m_s": 0.0,
        "source_hash": hash_json("source"),
        "model_hash": hash_json("model"),
        "asset_hash": hash_json("asset"),
        "gain_hash": hash_json("gain"),
        "joint_map_hash": hash_json("joint_map"),
    }
    report["report_hash"] = hash_json(report)
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")


def test_one_episode_is_one_sample_not_ten_frames(tmp_path: Path) -> None:
    folder = tmp_path / "episode"
    _fixture(folder)
    result = import_first_touch_curriculum((folder,))
    assert result["independent_physical_episode_count"] == 1
    assert result["clean_foot_only_episode_count"] == 1
    assert result["positive_foot_sides"] == ["left"]
    assert result["imitation_training_authorized"] is False
    assert result["episodes"][0]["precontact_frame_count"] == 1


def test_duplicate_and_tamper_are_rejected(tmp_path: Path) -> None:
    folder = tmp_path / "episode"
    _fixture(folder)
    with pytest.raises(ValueError, match="unique Isaac"):
        import_first_touch_curriculum((folder, folder))
    with (folder / "trajectory.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="unqualified"):
        import_first_touch_curriculum((folder,))


def test_lateral_teacher_cannot_enter_frozen_baseline_bank(tmp_path: Path) -> None:
    folder = tmp_path / "episode"
    _fixture(folder)
    path = folder / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["lateral_command_m_s"] = 0.08
    report.pop("report_hash")
    report["report_hash"] = hash_json(report)
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="unqualified"):
        import_first_touch_curriculum((folder,))
