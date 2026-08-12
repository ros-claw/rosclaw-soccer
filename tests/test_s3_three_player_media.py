from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.evidence.three_player import (
    _validate_request,
    load_three_player_trajectory,
)
from rosclaw_soccer.evidence.three_player import (
    validate_three_player_evidence as validate_bundle,
)
from rosclaw_soccer.media.three_player_video import (
    _goal_contract,
    _timelines,
    render_three_player_showcase_video,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    return source


def test_three_player_video_rejects_unknown_resolution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="resolution must be"):
        render_three_player_showcase_video(
            evidence_path=tmp_path / "evidence.json",
            asset_root=tmp_path / "assets",
            output_path=tmp_path / "video.mp4",
            source_checkout=tmp_path / "source",
            resolution="4k",
        )


def test_three_player_bundle_rejects_raw_evidence_inside_source(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="raw evidence must be outside"):
        validate_bundle(source / "evidence.json", source_checkout=source)


def test_three_player_trajectory_is_no_pickle_and_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.npz"
    values: dict[str, np.ndarray] = {
        "time": np.asarray((0.0, 1.0)),
        "ball_pose": np.asarray(((0, 0, 0.1, 1, 0, 0, 0),) * 2, dtype=float),
    }
    for role in ("passer", "shooter", "goalkeeper"):
        values[f"{role}_pelvis_pose"] = np.asarray(((0, 0, 0.8, 1, 0, 0, 0),) * 2, dtype=float)
        values[f"{role}_joint_position"] = np.zeros((2, 29))
    np.savez_compressed(path, **values)
    assert len(load_three_player_trajectory(path)["time"]) == 2

    values["goalkeeper_pelvis_pose"][1, 0] = np.nan
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="non-finite"):
        load_three_player_trajectory(path)


def test_three_player_metrics_reject_non_physical_pass_speed_jump(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = _source(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    report = {
        "request_hash": "sha256:request",
        "trajectory_hash": "sha256:trajectory",
        "trajectory_digest": "sha256:digest",
        "passed": True,
        "strict_replay": True,
        "simultaneous_three_body_physics": True,
        "shared_ball_state": True,
        "unified_physics_and_render_scene": True,
        "activation_ceiling": "SIM_ONLY",
        "physics_authority": "CPU_MUJOCO",
        "hardware_command_sent": False,
        "claims": {"real_hardware": False, "pixels_used_for_promotion": False},
        "pass_distance_m": 3.0,
        "shot_distance_m": 6.5,
        "pass_speed_max_positive_step_mps": 0.031,
        "result": {
            "passed": True,
            "finite_state": True,
            "pass_contact_observed": True,
            "shot_contact_observed": True,
            "goal_crossed": True,
            "goalkeeper_enabled": True,
            "pass_precision_passed": True,
            "joint_limit_violation": False,
            "torque_limit_violation": False,
            "actuator_saturation": False,
            "passer_post_kick_fall": False,
            "shooter_post_kick_fall": False,
            "goalkeeper_joint_limit_violation": False,
            "pass_delivery_error_m": 0.03,
            "target_error_m": 0.02,
            "goalkeeper_lateral_displacement_m": 1.0,
            "goalkeeper_min_pelvis_height_m": 0.75,
            "pass_contact_time_sec": 1.0,
            "shot_contact_time_sec": 2.0,
        },
    }
    (outside / "evidence.json").write_text(json.dumps(report), encoding="utf-8")
    (outside / "request.json").write_text("{}", encoding="utf-8")
    (outside / "trajectory.npz").write_bytes(b"placeholder")
    monkeypatch.setattr(
        "rosclaw_soccer.evidence.three_player._file_hash",
        lambda path: "sha256:request" if path.name == "request.json" else "sha256:trajectory",
    )
    monkeypatch.setattr(
        "rosclaw_soccer.evidence.three_player.load_three_player_trajectory", lambda _path: {}
    )
    monkeypatch.setattr(
        "rosclaw_soccer.evidence.three_player.trajectory_digest", lambda _value: "sha256:digest"
    )
    with pytest.raises(ValueError, match="positive jump"):
        validate_bundle(outside / "evidence.json", source_checkout=source)


def test_three_player_timeline_keeps_continuous_and_recovery_clips(tmp_path: Path) -> None:
    bundle = pytest.importorskip("rosclaw_soccer.evidence.three_player").ThreePlayerEvidenceBundle(
        evidence_path=tmp_path / "evidence.json",
        request_path=tmp_path / "request.json",
        trajectory_path=tmp_path / "trajectory.npz",
        report={"result": {"pass_contact_time_sec": 5.6, "shot_contact_time_sec": 7.92}},
        request={},
        trajectory={"time": np.linspace(0.02, 15.0, 750)},
        evidence_hash="sha256:evidence",
        request_hash="sha256:request",
        trajectory_hash="sha256:trajectory",
        trajectory_digest="sha256:digest",
    )
    timelines, clips = _timelines(bundle, 30)
    assert clips[1].title == "PASS → FINISH → RECOVERY"
    assert clips[4].title == "SHOOTER RECOVERY"
    assert clips[5].title == "PASSER RECOVERY"
    assert len(timelines[1]) >= 440


def test_three_player_goal_contract_is_derived_from_geometry() -> None:
    assert (
        _goal_contract(G1TrainingGoalSpec(width_m=7.32, height_m=2.44))
        == "REGULATION_7.32X2.44M_GOAL"
    )
    assert (
        _goal_contract(G1TrainingGoalSpec(width_m=3.0, height_m=2.0)) == "TRAINING_3.00X2.00M_GOAL"
    )


def test_three_player_request_rejects_target_outside_goal_contract() -> None:
    request = {
        "activation_ceiling": "SIM_ONLY",
        "physics_authority": "CPU_MUJOCO",
        "body_hash": "sha256:body",
        "goal_spec": {"plane_x_m": 7.5, "target_y_m": 0.89, "target_z_m": 0.115},
        "physical_scoring_target_m": [7.5, 0.89, 0.215],
    }
    with pytest.raises(ValueError, match="does not match the goal contract"):
        _validate_request(request, {"body_hash": "sha256:body"})
