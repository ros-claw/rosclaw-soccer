"""Foot geometry reports remain authenticated observation, not motor authority."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.foot_geometry_diagnostics import RIGHT_KNEE_INDEX, mine_foot_geometry
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def test_mine_foot_geometry_identifies_shin_first_without_causal_claim(tmp_path: Path) -> None:
    ball = np.zeros((50, 2, 3))
    ball[:, 0] = (2.4, -0.1, 0.11)
    ball[:, 1] = (2.6, 7.9, 0.11)
    force = np.zeros((50, 2, 6))
    force[40, 0, 5] = 10.0
    force[41, 1, 0] = 10.0
    trace = tmp_path / "trace.npz"
    np.savez_compressed(
        trace,
        ball_position_m=ball,
        ball_angular_velocity_rad_s=np.zeros((50, 2, 3)),
        ball_body_contact_force_peak_n=force,
    )
    joints = np.zeros((50, 2, 29))
    joints[:, 0, RIGHT_KNEE_INDEX] = np.linspace(0.3, 1.0, 50)
    feet = np.zeros((50, 2, 4, 3))
    feet[:, 0, 1, 0] = 2.0 - joints[:, 0, RIGHT_KNEE_INDEX] * 0.1
    feet[:, 0, 3, 0] = 2.0
    body_trace = tmp_path / "body_trace.npz"
    np.savez_compressed(
        body_trace,
        root_pose_xyzw_m=np.zeros((50, 2, 7)),
        root_velocity_world=np.zeros((50, 2, 6)),
        joint_position_rad=joints,
        joint_velocity_rad_s=np.zeros((50, 2, 29)),
        joint_target_rad=np.zeros((50, 2, 29)),
        navigation_speed_mps=np.full((50, 2), 1.4),
        ball_position_before_step_m=ball,
        ball_linear_velocity_before_step_m_s=np.zeros((50, 2, 3)),
        foot_geometry_position_before_step_m=feet,
        foot_geometry_velocity_before_step_m_s=np.zeros((50, 2, 4, 3)),
    )
    report = {
        "schema": "rsi_isaac_vector_first_touch_smoke_v1",
        "activation_ceiling": "SIM_ONLY",
        "learning_authorized": False,
        "promotion_authorized": False,
        "frames": 50,
        "source_hash": hash_json("runner"),
        "asset_hash": hash_json("asset"),
        "sonic_qualification_hash": hash_json("sonic"),
        "trace_hash": hash_bytes(trace.read_bytes()),
        "body_trace_hash": hash_bytes(body_trace.read_bytes()),
        "foot_geometry_body_names": [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_knee_link",
            "right_knee_link",
        ],
        "environments": [
            {
                "environment": i,
                "lane_y_m": i * 8.0,
                "course": {
                    "ball_x_m": (2.4, 2.6)[i],
                    "ball_y_local_m": -0.1,
                    "ball_vx_m_s": (-0.5, 0.5)[i],
                },
                "minimum_pelvis_z_m": 0.7,
                "first_contact_frame": (40, 41)[i],
                "contact_body_indices": ([5], [0])[i],
                "ball_final_local_xyz_m": [(2.4, 2.6)[i], -0.1, 0.11],
            }
            for i in range(2)
        ],
    }
    report["report_hash"] = hash_json(report)
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    result = mine_foot_geometry(tmp_path)
    assert result["incoming_contact_count"] == 1
    assert result["incoming_foot_behind_knee_count"] == 1
    assert result["incoming_first_nonfoot_count"] == 1
    assert result["episodes"][0]["precontact_knee_vs_foot_forward_correlation"] < -0.99
    assert result["causal_claim_authorized"] is False
