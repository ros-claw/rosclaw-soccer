"""Training isolation and safety bounds for the contextual contact actor."""

import json
from pathlib import Path

import pytest
from rsi_sonic_contextual_aim_actor import (
    RESERVED,
    TRAIN_X,
    TRAIN_Y,
    _action,
    _read_verified_report,
    _train_score,
)

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def test_new_reserved_courses_do_not_overlap_training_grid() -> None:
    assert len(RESERVED) == 3
    assert not set(RESERVED) & {(x, y) for x in TRAIN_X for y in TRAIN_Y}


def test_contextual_actor_interpolates_and_rejects_outside_support() -> None:
    anchors = [
        {"ball_y_m": y, "action": [0.48, 0.18, 0.1 * index, 0.02 * index]}
        for index, y in enumerate(TRAIN_Y)
    ]
    assert _action(anchors, 0.12) == pytest.approx((0.48, 0.18, 0.15, 0.03))
    with pytest.raises(ValueError, match="covered"):
        _action(anchors, 0.2)
    anchors[0]["action"][2] = 2.0
    with pytest.raises(ValueError, match="safety envelope"):
        _action(anchors, 0.06)


def test_missed_goal_has_dense_but_lower_reward() -> None:
    report = {
        "minimum_pelvis_height_m": 0.7,
        "peak_pelvis_tilt_rad": 0.2,
        "actuator_saturation_fraction": 0.0,
        "first_robot_ball_contact_is_foot": True,
        "goal_crossing_ball_center_xyz_m": None,
        "final_ball_displacement_xyz_m": [2.0, 0.1, 0.0],
        "ball_initial_xy_m": [2.15, 0.1],
        "peak_ball_speed_mps": 3.0,
    }
    nearer = _train_score(report)
    farther = _train_score({**report, "final_ball_displacement_xyz_m": [0.5, 1.0, 0.0]})
    goal = _train_score({**report, "goal_crossing_ball_center_xyz_m": [5.11, 0.1, 0.11]})
    assert goal > nearer > farther
    assert _train_score({**report, "minimum_pelvis_height_m": 0.5}) == -10.0


def test_verified_report_rejects_tampered_trajectory(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.npz"
    trajectory.write_bytes(b"trajectory")
    report = {"trajectory_hash": hash_bytes(trajectory.read_bytes())}
    report["report_hash"] = hash_json(report)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert _read_verified_report(path)["trajectory_hash"] == report["trajectory_hash"]
    trajectory.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="trajectory integrity"):
        _read_verified_report(path)
