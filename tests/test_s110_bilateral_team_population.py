from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.media.bilateral_team_population_video import (
    _implementation_hash as video_implementation_hash,
)
from rosclaw_soccer.media.bilateral_team_population_video import (
    validate_bilateral_team_population_video,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import G1SharedWorldResult
from rosclaw_soccer.training.bilateral_continuous_team_population import (
    BilateralContinuousTeamCase,
    BilateralContinuousTeamPopulationConfig,
    _perturbed_goal,
    attribute_population_failure,
    validate_bilateral_continuous_team_population,
)
from rosclaw_soccer.training.continuous_second_striker_save_exam import _continuous_metrics
from rosclaw_soccer.world.field import G1TrainingGoalSpec


def test_default_population_covers_feet_lanes_mass_and_friction() -> None:
    config = BilateralContinuousTeamPopulationConfig()

    assert len(config.cases) == 4
    assert {case.striker.kick_foot for case in config.cases} == {"left", "right"}
    assert len({case.lane_id for case in config.cases}) == 2
    assert len({case.ball_mass_kg for case in config.cases}) == 4
    assert len({case.ball_ground_friction for case in config.cases}) == 4
    assert config.config_hash.startswith("sha256:")
    with pytest.raises(ValueError, match="population is incomplete"):
        BilateralContinuousTeamPopulationConfig(cases=config.cases[:3])


def test_case_validation_is_fail_closed() -> None:
    case = BilateralContinuousTeamPopulationConfig().cases[0]

    with pytest.raises(ValueError, match="case is invalid"):
        BilateralContinuousTeamCase(
            case_id="unsafe/slash",
            lane_id=case.lane_id,
            striker=case.striker,
        )
    with pytest.raises(ValueError, match="case is invalid"):
        BilateralContinuousTeamCase(
            case_id="mass",
            lane_id=case.lane_id,
            striker=case.striker,
            ball_mass_kg=1.0,
        )


def test_ground_perturbation_does_not_corrupt_foot_ball_contact_material() -> None:
    case = BilateralContinuousTeamPopulationConfig().cases[1]
    goal = G1TrainingGoalSpec()

    perturbed = _perturbed_goal(goal, case)

    assert perturbed.ball_mass_kg == case.ball_mass_kg
    assert perturbed.ball_contact_sliding_friction == goal.ball_contact_sliding_friction
    assert case.ball_ground_friction != goal.ball_sliding_friction


def test_failure_attribution_keeps_earliest_phase_and_all_counterexamples() -> None:
    gates = {
        "qualified_first_airborne_save": True,
        "four_g1_two_ball_from_time_zero": True,
        "measured_ready_rearm_before_foot_contact": True,
        "stationary_second_ball_until_contact": True,
        "anatomical_second_striker_contact": True,
        "learned_multi_role_contact_stack_active": True,
        "bounded_forward_high_launch": False,
        "new_causal_goalkeeper_flight_epoch": False,
        "collision_faithful_high_glove_contact": False,
        "outward_physical_save": False,
        "second_striker_remains_stable": False,
        "whole_world_safety": False,
        "continuous_clock": True,
        "final_goalkeeper_ready": False,
    }

    attributed = attribute_population_failure({"passed": False, "gates": gates})

    assert attributed["passed"] is False
    assert attributed["first_failed_phase"] == "second_striker_contact"
    assert attributed["learning_owner"] == "second_striker"
    assert "bounded_forward_high_launch" in attributed["failed_gates"]
    assert "outward_physical_save" in attributed["failed_gates"]
    assert attributed["phases"][0]["passed"] is True


def test_missing_downstream_glove_does_not_erase_contact_phase_credit() -> None:
    time = np.arange(100, dtype=np.float64) * 0.02
    actor_active = np.zeros(100, dtype=np.bool_)
    actor_active[48:50] = True
    target_active = np.zeros(100, dtype=np.bool_)
    target_active[45:55] = True
    torque_active = np.zeros(100, dtype=np.bool_)
    torque_active[47:52] = True
    actor_torque = np.zeros((100, 29), dtype=np.float64)
    actor_torque[48, 6] = 12.0
    trajectory = {
        "time": time,
        "second_ball_velocity": np.zeros((100, 6), dtype=np.float64),
        "second_striker_ballistic_actor_active": actor_active,
        "second_striker_ballistic_actor_torque": actor_torque,
        "second_striker_ballistic_contact_active": target_active,
        "second_striker_ballistic_contact_torque_active": torque_active,
        "goalkeeper_observed_flight_active": np.zeros(100, dtype=np.bool_),
        "goalkeeper_observed_flight_start_sec": np.full(100, np.nan),
        "goalkeeper_reaction_active": np.zeros(100, dtype=np.bool_),
    }
    result = G1SharedWorldResult(
        finite_state=True,
        pass_contact_observed=True,
        shot_contact_observed=True,
        pass_contact_time_sec=0.2,
        shot_contact_time_sec=0.6,
        pass_peak_ball_speed_mps=1.0,
        shot_peak_ball_speed_mps=8.0,
        goal_crossed=False,
        goal_plane_crossed=False,
        goal_crossing_y_m=None,
        goal_crossing_z_m=None,
        target_error_m=None,
        passer_min_pelvis_height_m=0.75,
        shooter_min_pelvis_height_m=0.75,
        passer_roll_peak_rad=0.0,
        passer_pitch_peak_rad=0.0,
        shooter_roll_peak_rad=0.0,
        shooter_pitch_peak_rad=0.0,
        passer_tail_wobble_index=0.0,
        shooter_tail_wobble_index=0.0,
        receiver_phase_hold_frames=0,
        receiver_phase_advance_frames=0,
        receiver_max_ball_phase_error_m=0.0,
        robot_robot_contact_count=0,
        joint_limit_violation=False,
        torque_limit_violation=False,
        actuator_saturation=False,
        physics_steps=1000,
        second_striker_contact_time_sec=1.0,
    )

    metrics = _continuous_metrics(result, trajectory)

    assert metrics["valid"] is True
    assert metrics["save_phase_valid"] is False
    assert metrics["reason"] == "second glove timestamp is absent"
    assert metrics["actor_active_frame_count"] == 2
    assert metrics["actor_peak_torque_nm"] == pytest.approx(12.0)
    assert metrics["contact_target_memory_active_frame_count"] == 10
    assert metrics["upper_corner_torque_memory_active_frame_count"] == 5


def _rejected_evidence(directory: Path) -> Path:
    cases: dict[str, object] = {}
    for index in range(4):
        trajectory = directory / f"case-{index}.npz"
        trajectory.write_bytes(f"trajectory-{index}".encode())
        cases[f"case-{index}"] = {
            "strict_replay": True,
            "trajectory_file": trajectory.name,
            "trajectory_hash": hash_bytes(trajectory.read_bytes()),
        }
    payload = {
        "schema_version": "rosclaw_soccer.bilateral_continuous_team_population_evidence.v1",
        "claim": "BILATERAL_PERTURBED_CONTINUOUS_TEAM_POPULATION",
        "passed": False,
        "promotion_status": "REJECTED_BILATERAL_POPULATION",
        "growth_status": "COUNTEREXAMPLES_RETAINED_FOR_GROWTH",
        "population_gates": {"all_cases_qualified": False},
        "passed_cases": ["case-0"],
        "rejected_cases": ["case-1", "case-2", "case-3"],
        "observed_contact_feet": ["left", "right"],
        "cases": cases,
        "request_hash": "sha256:request",
        "source_commit": "a" * 40,
        "implementation_hash": "sha256:implementation",
        "physics_backend": "mujoco_cpu",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "reset_or_teleport_used": False,
        "ball_cannon_used": False,
        "pixels_used_for_scoring": False,
    }
    payload["report_hash"] = hash_json(payload)
    evidence = directory / "evidence.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    return evidence


def test_rejected_population_is_valid_growth_input_but_not_promotion(tmp_path: Path) -> None:
    evidence = _rejected_evidence(tmp_path)

    report = validate_bilateral_continuous_team_population(evidence)

    assert report["passed"] is False
    assert report["promotion_status"] == "REJECTED_BILATERAL_POPULATION"
    assert report["growth_status"] == "COUNTEREXAMPLES_RETAINED_FOR_GROWTH"

    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["hardware_command_sent"] = True
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authority or integrity"):
        validate_bilateral_continuous_team_population(evidence)


def test_population_request_is_content_bound() -> None:
    config = BilateralContinuousTeamPopulationConfig()

    assert asdict(config)["cases"][2]["striker"]["kick_foot"] == "left"
    assert asdict(config)["exam"]["activation_ceiling"] == "SIM_ONLY"


def test_rejected_video_manifest_cannot_claim_promotion(tmp_path: Path) -> None:
    video = tmp_path / "diagnostic.mp4"
    source = tmp_path / "evidence.json"
    video.write_bytes(b"video")
    source.write_bytes(b"evidence")
    payload = {
        "schema_version": "rosclaw_soccer.bilateral_team_population_video.v1",
        "video_path": str(video),
        "video_hash": hash_bytes(video.read_bytes()),
        "source_files": {str(source): hash_bytes(source.read_bytes())},
        "claim": "BILATERAL_PERTURBED_CONTINUOUS_TEAM_POPULATION_DIAGNOSTIC",
        "truth_label": "REJECTED_DEVELOPMENT",
        "evidence_report_hash": "sha256:evidence",
        "evidence_passed": False,
        "strict_replay_all_cases": True,
        "observed_contact_feet": ["left", "right"],
        "fps": 60,
        "width": 1920,
        "height": 1080,
        "frame_count": 600,
        "duration_sec": 10.0,
        "clips": [],
        "visualization_only": True,
        "pixels_used_for_scoring": False,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "implementation_hash": video_implementation_hash(),
    }
    payload["manifest_hash"] = hash_json(payload)
    manifest = tmp_path / "diagnostic.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert validate_bilateral_team_population_video(manifest)["evidence_passed"] is False

    payload["promotion_eligible"] = True
    payload["manifest_hash"] = hash_json(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authority contract"):
        validate_bilateral_team_population_video(manifest)
