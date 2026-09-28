"""Vector contact evidence must count physical episodes, not control frames."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses
from rosclaw_soccer.rsi.vector_first_touch_evidence import (
    audit_first_touch_candidate_execution,
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
        "asset_hash": hash_json("asset"),
        "sonic_qualification_hash": hash_json("sonic"),
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


def test_vector_audit_requires_bounded_navigation_speed(tmp_path: Path) -> None:
    folder = tmp_path / "case"
    _fixture(folder)
    report_path = folder / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["navigation_speed_mps"] = 1.2
    report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert audit_vector_first_touch(folder)["clean_foot_only_episode_count"] == 1
    report["navigation_speed_mps"] = 1.6
    report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="unauthenticated"):
        audit_vector_first_touch(folder)
    report["navigation_speed_mps"] = float("nan")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        audit_vector_first_touch(folder)


def test_vector_audit_requires_complete_bounded_near_ball_schedule(tmp_path: Path) -> None:
    folder = tmp_path / "case"
    _fixture(folder)
    report_path = folder / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(
        near_ball_gap_m=1.4,
        near_ball_speed_mps=0.8,
        near_ball_incoming_only=True,
    )
    report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert audit_vector_first_touch(folder)["clean_foot_only_episode_count"] == 1
    report["near_ball_incoming_only"] = "yes"
    report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="unauthenticated"):
        audit_vector_first_touch(folder)


def test_vector_audit_commits_body_and_command_trace(tmp_path: Path) -> None:
    folder = tmp_path / "case"
    _fixture(folder)
    report_path = folder / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(near_ball_gap_m=1.4, near_ball_speed_mps=0.8, near_ball_incoming_only=True)
    speeds = np.full((50, 2), 1.4)
    speeds[20:30, 0] = 0.8
    arrays = {
        "root_pose_xyzw_m": np.zeros((50, 2, 7)),
        "root_velocity_world": np.zeros((50, 2, 6)),
        "joint_position_rad": np.zeros((50, 2, 29)),
        "joint_velocity_rad_s": np.zeros((50, 2, 29)),
        "joint_target_rad": np.zeros((50, 2, 29)),
        "navigation_speed_mps": speeds,
    }
    path = folder / "body_trace.npz"
    np.savez_compressed(path, **arrays)
    report["body_trace_hash"] = hash_bytes(path.read_bytes())
    report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
    report_path.write_text(json.dumps(report), encoding="utf-8")
    audited = audit_vector_first_touch(folder)
    assert audited["recorded_body_frames"] == 50
    assert audited["changed_navigation_command_frames"] == 10
    arrays["navigation_speed_mps"][0, 0] = 0.7
    np.savez_compressed(path, **arrays)
    report["body_trace_hash"] = hash_bytes(path.read_bytes())
    report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="uncommitted navigation speed"):
        audit_vector_first_touch(folder)


def test_vector_audit_requires_bounded_batched_torch_shadow(tmp_path: Path) -> None:
    folder = tmp_path / "case"
    _fixture(folder)
    report_path = folder / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["torch_batch_shadow"] = True
    report["torch_batch_drive"] = False
    report["torch_batch_max_target_difference_rad"] = 1e-6
    report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert audit_vector_first_touch(folder)["independent_physical_episode_count"] == 2
    report["torch_batch_max_target_difference_rad"] = 0.002
    report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="unauthenticated"):
        audit_vector_first_touch(folder)


def test_vector_audit_requires_bounded_plan_only_target(tmp_path: Path) -> None:
    folder = tmp_path / "case"
    _fixture(folder)
    report_path = folder / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["torch_batch_plan_only"] = True
    report["torch_batch_max_internal_target_difference_rad"] = 1e-6
    report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert audit_vector_first_touch(folder)["independent_physical_episode_count"] == 2
    report["torch_batch_max_internal_target_difference_rad"] = 0.002
    report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="unauthenticated"):
        audit_vector_first_touch(folder)


def test_seeded_sixteen_course_split_is_recomputed(tmp_path: Path) -> None:
    folder = tmp_path / "case"
    _fixture(folder)
    report_path = folder / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    courses = sample_training_courses(20260928)
    positions = np.zeros((50, 16, 3))
    entries = []
    for i, (x, y, vx) in enumerate(courses):
        positions[:, i] = (x, i * 8.0 + y, 0.11)
        entries.append(
            {
                "environment": i,
                "lane_y_m": i * 8.0,
                "course": {"ball_x_m": x, "ball_y_local_m": y, "ball_vx_m_s": vx},
                "minimum_pelvis_z_m": 0.7,
                "first_contact_frame": None,
                "contact_body_indices": [],
                "ball_final_local_xyz_m": [x, y, 0.11],
            }
        )
    trace = folder / "trace.npz"
    np.savez_compressed(
        trace,
        ball_position_m=positions,
        ball_angular_velocity_rad_s=np.zeros_like(positions),
        ball_body_contact_force_peak_n=np.zeros((50, 16, 6)),
    )
    report["trace_hash"] = hash_bytes(trace.read_bytes())
    report["environments"] = entries
    report["training_course_seed"] = 20260928
    report["course_catalog_hash"] = hash_json(courses)
    report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert audit_vector_first_touch(folder)["independent_physical_episode_count"] == 16
    report["environments"][0]["course"]["ball_x_m"] += 0.001
    report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="training courses differ"):
        audit_vector_first_touch(folder)


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


def test_candidate_audit_recomputes_foot_only_reward(tmp_path: Path) -> None:
    from rosclaw_soccer.rsi.first_touch_candidate import JOINT_NAMES, candidate_manifest

    parent_folder, folder = tmp_path / "parent", tmp_path / "candidate"
    _fixture(parent_folder)
    folder.mkdir()
    parent = json.loads((parent_folder / "report.json").read_text(encoding="utf-8"))
    courses = tuple(
        (
            row["course"]["ball_x_m"],
            row["course"]["ball_y_local_m"],
            row["course"]["ball_vx_m_s"],
        )
        for row in parent["environments"]
    )
    actions = tuple((0.0,) * len(JOINT_NAMES) for _ in courses)
    manifest = candidate_manifest(
        courses=courses, parent_report_hash=parent["report_hash"], actions_rad=actions, seed=1
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with np.load(parent_folder / "trace.npz", allow_pickle=False) as original:
        np.savez_compressed(folder / "trace.npz", **{k: original[k] for k in original.files})
    report = {
        **parent,
        "schema": "rsi_isaac_vector_first_touch_candidate_execution_v2",
        "trace_hash": hash_bytes((folder / "trace.npz").read_bytes()),
        "parent_report_hash": parent["report_hash"],
        "candidate_hash": manifest["candidate_hash"],
        "candidate_action_joint_names": list(JOINT_NAMES),
        "candidate_actions_rad": [list(row) for row in actions],
        "candidate_actions_applied_frames": [1, 1],
        "candidate_action_projection_count": [0, 0],
        "trained_actor": False,
    }
    report.pop("report_hash")
    report["report_hash"] = hash_json(report)
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
    result = audit_first_touch_candidate_execution(
        folder, parent_folder=parent_folder, candidate_path=manifest_path
    )
    assert result["candidate_clean_foot_only_count"] == 1
    assert result["reward_per_course"] == [1.0, -0.5]
    assert result["fresh_opened"] is False
    report["training_course_seed"] = 20260928
    report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
    (folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="unauthenticated"):
        audit_first_touch_candidate_execution(
            folder, parent_folder=parent_folder, candidate_path=manifest_path
        )
