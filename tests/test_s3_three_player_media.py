from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.evidence.three_player import (
    _validate_agility_metrics,
    _validate_composite_imitation_metrics,
    _validate_follow_through_metrics,
    _validate_imitation_metrics,
    _validate_implementation,
    _validate_metrics,
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
        "ball_velocity": np.zeros((2, 6), dtype=float),
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


def test_three_player_development_candidate_needs_anticipation() -> None:
    result = {
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
        "target_error_m": 0.06,
        "goalkeeper_lateral_displacement_m": 0.25,
        "goalkeeper_min_pelvis_height_m": 0.75,
        "goalkeeper_ball_contact_observed": False,
        "goalkeeper_anticipation_active_fraction": 0.032,
        "pass_contact_time_sec": 5.0,
        "shot_contact_time_sec": 7.5,
    }
    report = {
        "pass_distance_m": 3.0,
        "shot_distance_m": 6.5,
        "pass_speed_max_positive_step_mps": 0.0,
        "result": result,
    }
    _validate_metrics(report, allow_development_candidate=True)
    with pytest.raises(ValueError, match="did not anticipate"):
        _validate_metrics(
            {**report, "result": {**result, "goalkeeper_anticipation_active_fraction": 0.0}},
            allow_development_candidate=True,
        )


def test_goalkeeper_block_has_its_own_success_semantics() -> None:
    result = {
        "passed": False,
        "finite_state": True,
        "pass_contact_observed": True,
        "shot_contact_observed": True,
        "goal_crossed": False,
        "goalkeeper_enabled": True,
        "pass_precision_passed": True,
        "goalkeeper_ball_contact_observed": True,
        "goalkeeper_save_observed": True,
        "joint_limit_violation": False,
        "torque_limit_violation": False,
        "actuator_saturation": False,
        "passer_post_kick_fall": False,
        "shooter_post_kick_fall": False,
        "goalkeeper_joint_limit_violation": False,
        "pass_delivery_error_m": 0.03,
        "goalkeeper_min_pelvis_height_m": 0.73,
        "pass_contact_time_sec": 5.0,
        "shot_contact_time_sec": 7.0,
        "goalkeeper_ball_contact_time_sec": 8.0,
    }
    report = {
        "schema_version": "rosclaw_soccer.goalkeeper_block_evidence.v1",
        "passed": True,
        "promotion_status": "PASSED_DEVELOPMENT_GATE_NOT_PROMOTED",
        "baseline_goal_crossed": True,
        "baseline_goalkeeper_contact_observed": False,
        "pass_distance_m": 3.0,
        "shot_to_block_distance_m": 5.2,
        "selected_policy_hash": "sha256:" + "a" * 64,
        "search": {
            "selected_trial": {
                "eligible": True,
                "safety_cost": 0.0,
                "post_contact_speed_ratio": 0.68,
                "policy_hash": "sha256:" + "a" * 64,
            }
        },
        "claims": {
            "goalkeeper_save_achieved": True,
            "candidate_promoted": False,
        },
        "result": result,
    }

    _validate_metrics(report)
    rejected = {
        **report,
        "passed": False,
        "promotion_status": "REJECTED_DEVELOPMENT",
        "baseline_goal_crossed": False,
        "baseline_goalkeeper_contact_observed": True,
    }
    _validate_metrics(rejected, allow_development_candidate=True)
    with pytest.raises(ValueError, match="did not pass"):
        _validate_metrics(rejected)
    with pytest.raises(ValueError, match="save or safety"):
        _validate_metrics({**report, "result": {**result, "goal_crossed": True}})


def test_composite_imitation_requires_train_only_contact_teacher() -> None:
    prior = "sha256:" + "a" * 64
    result = {
        "passed": True,
        "finite_state": True,
        "pass_contact_observed": True,
        "shot_contact_observed": True,
        "goal_crossed": True,
        "goalkeeper_enabled": True,
        "pass_precision_passed": True,
        "target_error_m": 0.02,
        "shooter_motion_prior_hash": prior,
        "shooter_contact_prior_hash": prior,
    }
    tracking = {
        "teacher_displacement_error_rms_rad": 0.69,
        "peak_contact_target_delta_rad": 0.001,
        "active_fraction": 0.02,
        "schema_version": "rosclaw_soccer.g1_contact_imitation_metrics.v1",
    }
    report = {
        "passed": True,
        "promotion_status": "PASSED_DEVELOPMENT_GATE_NOT_PROMOTED",
        "motion_prior_hash": prior,
        "contact_prior_hash": prior,
        "selected_candidate_hash": prior,
        "candidate_contact_tracking": tracking,
        "search": {
            "selected_trial": {
                "eligible": True,
                "candidate_hash": prior,
                "omnicontact_tracking": tracking,
            }
        },
        "claims": {
            "omnicontact_train_only_contact_teacher": True,
            "omnicontact_heldout_metrics_accessed": False,
            "teacher_direct_torque_output": False,
            "candidate_promoted": False,
        },
        "result": result,
    }

    _validate_composite_imitation_metrics(report)
    with pytest.raises(ValueError, match="held-out"):
        _validate_composite_imitation_metrics(
            {
                **report,
                "claims": {**report["claims"], "omnicontact_heldout_metrics_accessed": True},
            }
        )


def test_imitation_evidence_binds_teacher_and_stability_plasticity_gate() -> None:
    prior_hash = "sha256:" + "b" * 64
    candidate_hash = "sha256:" + "c" * 64
    result = {
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
        "target_error_m": 0.02,
        "shooter_motion_prior_hash": prior_hash,
    }
    report = {
        "passed": True,
        "promotion_status": "PASSED_DEVELOPMENT_GATE_NOT_PROMOTED",
        "motion_prior_hash": prior_hash,
        "selected_candidate_hash": candidate_hash,
        "pass_speed_max_positive_step_mps": 0.02,
        "search": {"selected_trial": {"eligible": True, "candidate_hash": candidate_hash}},
        "claims": {
            "motiondecode_whole_body_position_teacher": True,
            "motiondecode_whole_body_velocity_teacher": True,
            "candidate_promoted": False,
        },
        "result": result,
    }

    _validate_imitation_metrics(report)
    with pytest.raises(ValueError, match="motion-prior hash"):
        _validate_imitation_metrics(
            {**report, "result": {**result, "shooter_motion_prior_hash": "sha256:wrong"}}
        )


def test_agility_evidence_binds_local_basin_and_stability_metrics() -> None:
    prior_hash = "sha256:" + "b" * 64
    contact_hash = "sha256:" + "d" * 64
    candidate_hash = "sha256:" + "c" * 64
    report = {
        "passed": True,
        "promotion_status": "PASSED_DEVELOPMENT_GATE_NOT_PROMOTED",
        "motion_prior_hash": prior_hash,
        "contact_prior_hash": contact_hash,
        "selected_candidate_hash": candidate_hash,
        "search": {
            "selected_trial": {"eligible": True, "candidate_hash": candidate_hash},
            "neighborhood": {"passed": True},
        },
        "candidate_naturalness": {
            "post_contact_support_slip_m": 0.05,
            "post_contact_peak_backward_velocity_mps": 0.002,
        },
        "claims": {
            "joint_group_position_velocity_authority_separated": True,
            "counterfactual_parent_retained": True,
            "local_neighborhood_gate_passed": True,
            "motiondecode_whole_body_teacher": True,
            "omnicontact_train_only_contact_teacher": True,
            "omnicontact_heldout_metrics_accessed": False,
            "teacher_direct_torque_output": False,
            "candidate_promoted": False,
        },
        "result": {
            "passed": True,
            "finite_state": True,
            "pass_contact_observed": True,
            "shot_contact_observed": True,
            "goal_crossed": True,
            "goalkeeper_enabled": True,
            "pass_precision_passed": True,
            "target_error_m": 0.009,
            "shooter_motion_prior_hash": prior_hash,
            "shooter_contact_prior_hash": contact_hash,
        },
    }

    _validate_agility_metrics(report)
    with pytest.raises(ValueError, match="local neighborhood"):
        _validate_agility_metrics(
            {
                **report,
                "search": {**report["search"], "neighborhood": {"passed": False}},
            }
        )


def test_three_player_development_evidence_binds_current_implementation(tmp_path: Path) -> None:
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="implementation source is missing"):
        _validate_implementation(
            {"schema_version": "rosclaw_soccer.three_role_development_evidence.v1"},
            source,
        )


def test_follow_through_evidence_requires_visible_plasticity_and_retention() -> None:
    motion = "sha256:motion"
    contact = "sha256:contact"
    mosaic = "sha256:mosaic"
    report = {
        "passed": True,
        "promotion_status": "PASSED_DEVELOPMENT_GATE_NOT_PROMOTED",
        "motion_prior_hash": motion,
        "contact_prior_hash": contact,
        "mosaic_prior_hash": mosaic,
        "selected_candidate_hash": "sha256:candidate",
        "search": {
            "neighborhood_eligible_fraction": 1.0,
            "selected_trial": {"eligible": True, "candidate_hash": "sha256:candidate"},
        },
        "candidate_naturalness": {
            "post_contact_support_slip_m": 0.047,
            "post_contact_peak_backward_velocity_mps": 0.003,
        },
        "parent_follow_through": {
            "arm_excursion_rms_rad": 0.10,
            "upper_body_motion_energy": 0.10,
        },
        "candidate_follow_through": {
            "arm_excursion_rms_rad": 0.12,
            "upper_body_motion_energy": 0.12,
        },
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
            "target_error_m": 0.009,
            "shooter_motion_prior_hash": motion,
            "shooter_contact_prior_hash": contact,
            "shooter_agility_prior_hash": mosaic,
        },
        "claims": {
            "semantic_mosaic_soccer_teacher": True,
            "endpoint_neutral_pose_residual": True,
            "arm_only_plasticity_boundary": True,
            "visible_plasticity_floor_passed": True,
            "counterfactual_parent_retained": True,
            "local_neighborhood_gate_passed": True,
            "teacher_direct_torque_output": False,
            "candidate_promoted": False,
        },
    }

    _validate_follow_through_metrics(report)
    with pytest.raises(ValueError, match="arm-excursion"):
        _validate_follow_through_metrics(
            {
                **report,
                "candidate_follow_through": {
                    "arm_excursion_rms_rad": 0.105,
                    "upper_body_motion_energy": 0.12,
                },
            }
        )


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


def test_frozen_three_player_evidence_is_rejected_as_sliding() -> None:
    source = Path("/code/rosclaw/phase8_evidence/g1-three-player-long-relay-v3-final")
    if not source.is_dir():
        pytest.skip("frozen three-player evidence is unavailable")
    checkout = Path(__file__).resolve().parents[1]

    with pytest.raises(ValueError, match="sliding rather than rolling"):
        validate_bundle(source / "g1-three-player-showcase.json", source_checkout=checkout)
