"""Temporal policy execution is bound to Parent, pre-step body and applied action."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses
from rosclaw_soccer.rsi.temporal_first_touch_policy import (
    FEATURE_NAMES,
    JOINT_NAMES,
    candidate_manifest,
)
from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_temporal_first_touch_execution
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def test_zero_temporal_policy_audits_full_body_and_ball_alignment(tmp_path: Path) -> None:
    seed = 20260930
    courses = sample_training_courses(seed)
    parent_folder, folder = tmp_path / "parent", tmp_path / "candidate"
    parent_folder.mkdir()
    folder.mkdir()
    frames, count = 50, len(courses)
    lanes = np.arange(count) * 8.0
    positions = np.zeros((frames, count, 3))
    root = np.zeros((frames, count, 7))
    root[:, :, 1] = lanes
    root[:, :, 2] = 0.793
    root[:, :, 6] = 1.0
    for i, (x, y, _vx) in enumerate(courses):
        positions[:, i] = (x, lanes[i] + y, 0.13)
    trace_arrays = {
        "ball_position_m": positions,
        "ball_angular_velocity_rad_s": np.zeros((frames, count, 3)),
        "ball_body_contact_force_peak_n": np.zeros((frames, count, 6)),
    }
    for source in (parent_folder, folder):
        np.savez_compressed(source / "trace.npz", **trace_arrays)
    entries = [
        {
            "environment": i,
            "lane_y_m": float(lanes[i]),
            "course": {"ball_x_m": x, "ball_y_local_m": y, "ball_vx_m_s": vx},
            "minimum_pelvis_z_m": 0.7,
            "first_contact_frame": None,
            "contact_body_indices": [],
            "ball_final_local_xyz_m": [x, y, 0.13],
        }
        for i, (x, y, vx) in enumerate(courses)
    ]
    parent = {
        "schema": "rsi_isaac_vector_first_touch_smoke_v1",
        "activation_ceiling": "SIM_ONLY",
        "learning_authorized": False,
        "promotion_authorized": False,
        "source_hash": "sha256:" + "a" * 64,
        "asset_hash": "sha256:" + "b" * 64,
        "sonic_qualification_hash": "sha256:" + "c" * 64,
        "trace_hash": hash_bytes((parent_folder / "trace.npz").read_bytes()),
        "frames": frames,
        "navigation_speed_mps": 1.4,
        "torch_batch_plan_only": True,
        "torch_batch_max_internal_target_difference_rad": 0.0,
        "training_course_seed": seed,
        "course_catalog_hash": hash_json(courses),
        "environments": entries,
    }
    parent["report_hash"] = hash_json(parent)
    (parent_folder / "report.json").write_text(json.dumps(parent), encoding="utf-8")
    weights = np.zeros((count, len(FEATURE_NAMES), len(JOINT_NAMES)))
    manifest = candidate_manifest(
        courses=courses,
        parent_report_hash=parent["report_hash"],
        weights_per_course=weights,
        seed=0,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    ball_velocity = np.zeros((frames, count, 3))
    for i, (_x, _y, vx) in enumerate(courses):
        ball_velocity[:, i, 0] = vx
    body = {
        "root_pose_xyzw_m": root,
        "root_velocity_world": np.zeros((frames, count, 6)),
        "joint_position_rad": np.zeros((frames, count, 29)),
        "joint_velocity_rad_s": np.zeros((frames, count, 29)),
        "joint_target_rad": np.zeros((frames, count, 29)),
        "navigation_speed_mps": np.full((frames, count), 1.4),
        "ball_position_before_step_m": positions,
        "ball_linear_velocity_before_step_m_s": ball_velocity,
        "baseline_joint_target_rad": np.zeros((frames, count, 29)),
        "applied_residual_rad": np.zeros((frames, count, 3)),
    }
    np.savez_compressed(folder / "body_trace.npz", **body)
    candidate = {
        **parent,
        "schema": "rsi_isaac_vector_first_touch_temporal_candidate_v1",
        "trace_hash": hash_bytes((folder / "trace.npz").read_bytes()),
        "body_trace_hash": hash_bytes((folder / "body_trace.npz").read_bytes()),
        "parent_report_hash": parent["report_hash"],
        "candidate_hash": manifest["candidate_hash"],
        "temporal_policy_joint_names": list(JOINT_NAMES),
        "temporal_policy_applied_frames": [0] * count,
        "temporal_policy_projection_count": [0] * count,
        "trained_actor": False,
    }
    candidate.pop("report_hash")
    candidate["report_hash"] = hash_json(candidate)
    (folder / "report.json").write_text(json.dumps(candidate), encoding="utf-8")
    result = audit_temporal_first_touch_execution(
        folder, parent_folder=parent_folder, candidate_path=manifest_path
    )
    assert result["candidate_clean_foot_only_count"] == 0
    assert result["temporal_policy_applied_frames"] == [0] * count
    assert result["recorded_body_frames"] == frames
