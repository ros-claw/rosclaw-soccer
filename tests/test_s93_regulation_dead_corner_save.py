from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw_soccer.media.regulation_dead_corner_video import (
    validate_regulation_dead_corner_video_manifest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import G1SharedWorldResult
from rosclaw_soccer.training.regulation_dead_corner_save import (
    RegulationDeadCornerConfig,
    _implementation_hash,
    evaluate_dead_corner_baseline,
    regulation_dead_corner_lane_kwargs,
    regulation_dead_corner_lanes,
    validate_regulation_dead_corner_evidence,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec


def _result(*, crossing_y_m: float = 3.36) -> G1SharedWorldResult:
    return G1SharedWorldResult(
        finite_state=True,
        pass_contact_observed=True,
        shot_contact_observed=True,
        pass_contact_time_sec=2.0,
        shot_contact_time_sec=7.4,
        pass_peak_ball_speed_mps=2.0,
        shot_peak_ball_speed_mps=10.0,
        goal_crossed=True,
        goal_plane_crossed=True,
        goal_crossing_y_m=crossing_y_m,
        goal_crossing_z_m=1.25,
        target_error_m=1.0,
        passer_min_pelvis_height_m=0.68,
        shooter_min_pelvis_height_m=0.65,
        passer_roll_peak_rad=0.0,
        passer_pitch_peak_rad=0.0,
        shooter_roll_peak_rad=0.0,
        shooter_pitch_peak_rad=0.0,
        passer_tail_wobble_index=0.0,
        shooter_tail_wobble_index=0.0,
        receiver_phase_hold_frames=1,
        receiver_phase_advance_frames=1,
        receiver_max_ball_phase_error_m=0.0,
        robot_robot_contact_count=0,
        joint_limit_violation=False,
        torque_limit_violation=False,
        actuator_saturation=False,
        physics_steps=1,
        pass_delivery_error_m=0.004,
        pass_delivery_lateral_error_m=0.002,
    )


def test_regulation_dead_corner_contract_is_bilateral_and_sim_only() -> None:
    config = RegulationDeadCornerConfig()
    lanes = regulation_dead_corner_lanes()

    assert tuple(lane.lane_id for lane in lanes) == ("left-post", "right-post")
    assert {lane.expected_glove_side for lane in lanes} == {"left", "right"}
    assert config.maximum_post_surface_clearance_m == pytest.approx(0.15)
    assert config.activation_ceiling == "SIM_ONLY"
    assert config.hardware_authorized is False
    assert config.commercial_use_allowed is False
    with pytest.raises(ValueError, match="non-commercial SIM_ONLY"):
        replace(config, commercial_use_allowed=True)


def test_dead_corner_builder_places_attackers_and_keeper_at_regulation_posts(
    tmp_path: Path,
) -> None:
    artifacts = [tmp_path / name for name in ("striker", "keeper", "gmt", "skill")]
    for artifact in artifacts:
        artifact.write_text("{}", encoding="utf-8")
    dive = tmp_path / "dive"
    dive.mkdir()

    left, right = (
        regulation_dead_corner_lane_kwargs(
            lane=lane,
            striker_actor_path=artifacts[0],
            goalkeeper_actor_path=artifacts[1],
            gmt_model_path=artifacts[2],
            gmt_skill_path=artifacts[3],
            dive_source_checkout=dive,
        )
        for lane in regulation_dead_corner_lanes()
    )

    assert left["shooter_origin"][1] == pytest.approx(-3.83)
    assert right["shooter_origin"][1] == pytest.approx(2.88)
    assert left["goalkeeper_config"].initial_lateral_position_m == pytest.approx(-3.05)
    assert right["goalkeeper_config"].initial_lateral_position_m == pytest.approx(2.84)
    assert left["goalkeeper_config"].regulation_goal_positioning_enabled is True
    assert left["goalkeeper_config"].joint_guard_impact_lead_sec == pytest.approx(0.08)
    assert right["goalkeeper_config"].joint_guard_boundary_kd == pytest.approx(20.0)
    assert left["goalkeeper_config"].balanced_dive_lower_body_scale == pytest.approx(0.83)
    assert right["goalkeeper_config"].balanced_dive_lower_body_scale == pytest.approx(0.86)
    assert left["goal_spec"].width_m == pytest.approx(7.32)
    assert right["goal_spec"].height_m == pytest.approx(2.44)


def test_unopposed_gate_requires_ball_surface_near_the_post() -> None:
    config = RegulationDeadCornerConfig()
    goal = G1TrainingGoalSpec(width_m=7.32, height_m=2.44, regulation_field_enabled=True)
    accepted = evaluate_dead_corner_baseline(result=_result(), goal=goal, config=config)
    rejected = evaluate_dead_corner_baseline(
        result=_result(crossing_y_m=3.10), goal=goal, config=config
    )

    assert accepted["passed"] is True
    assert accepted["post_surface_clearance_m"] == pytest.approx(0.15)
    assert rejected["passed"] is False
    assert rejected["gates"]["regulation_post_dead_corner"] is False


def test_dead_corner_evidence_is_bound_to_baseline_and_save_trajectories(
    tmp_path: Path,
) -> None:
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    cases: dict[str, object] = {}
    for lane in ("left-post", "right-post"):
        case: dict[str, object] = {"passed": True}
        for prefix in ("baseline", "save"):
            trajectory = tmp_path / f"{lane}-{prefix}.npz"
            trajectory.write_bytes(f"{lane}-{prefix}".encode())
            case[f"{prefix}_trajectory_file"] = trajectory.name
            case[f"{prefix}_trajectory_hash"] = hash_bytes(trajectory.read_bytes())
        cases[lane] = case
    payload = {
        "schema_version": "rosclaw_soccer.regulation_dead_corner_evidence.v1",
        "passed": True,
        "promotion_status": "FROZEN_RESEARCH_DEMO",
        "claim": "STRICT_REGULATION_LATERAL_DEAD_CORNER_SAVE_PAIR",
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "pixels_used_for_scoring": False,
        "portfolio_gates": {"all": True},
        "request_hash": hash_bytes(request.read_bytes()),
        "cases": cases,
    }
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_regulation_dead_corner_evidence(evidence)["passed"] is True
    (tmp_path / "right-post-save.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="binding changed"):
        validate_regulation_dead_corner_evidence(evidence)


def test_dead_corner_video_manifest_is_visualization_only_and_hash_bound(
    tmp_path: Path,
) -> None:
    video = tmp_path / "showcase.mp4"
    source = tmp_path / "evidence.json"
    video.write_bytes(b"video")
    source.write_bytes(b"evidence")
    payload = {
        "schema_version": "rosclaw_soccer.regulation_dead_corner_video.v1",
        "video_path": str(video),
        "video_hash": hash_bytes(video.read_bytes()),
        "source_files": {str(source): hash_bytes(source.read_bytes())},
        "claim": "STRICT_REGULATION_LATERAL_DEAD_CORNER_SAVE_PAIR",
        "case_count": 2,
        "strict_replay": True,
        "fps": 60,
        "width": 1920,
        "height": 1080,
        "frame_count": 120,
        "duration_sec": 2.0,
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
    }
    payload["manifest_hash"] = hash_json(payload)
    manifest = tmp_path / "showcase.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_regulation_dead_corner_video_manifest(manifest)["case_count"] == 2
    video.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="video hash changed"):
        validate_regulation_dead_corner_video_manifest(manifest)
