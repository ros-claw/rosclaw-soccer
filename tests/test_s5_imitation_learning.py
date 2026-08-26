from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.skills.team.imitation_learning import (
    G1ImitationCandidate,
    G1MotionNaturalnessMetrics,
    evaluate_g1_imitation_trial,
)
from rosclaw_soccer.skills.team.shared_world import G1SharedWorldResult


def _hash(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _result(**overrides: object) -> G1SharedWorldResult:
    values: dict[str, object] = {
        "finite_state": True,
        "pass_contact_observed": True,
        "shot_contact_observed": True,
        "pass_contact_time_sec": 1.0,
        "shot_contact_time_sec": 2.0,
        "pass_peak_ball_speed_mps": 1.0,
        "shot_peak_ball_speed_mps": 8.0,
        "goal_crossed": True,
        "goal_plane_crossed": True,
        "goal_crossing_y_m": 0.89,
        "goal_crossing_z_m": 0.115,
        "target_error_m": 0.02,
        "passer_min_pelvis_height_m": 0.75,
        "shooter_min_pelvis_height_m": 0.75,
        "passer_roll_peak_rad": 0.1,
        "passer_pitch_peak_rad": 0.1,
        "shooter_roll_peak_rad": 0.2,
        "shooter_pitch_peak_rad": 0.2,
        "passer_tail_wobble_index": 0.01,
        "shooter_tail_wobble_index": 0.01,
        "receiver_phase_hold_frames": 0,
        "receiver_phase_advance_frames": 0,
        "receiver_max_ball_phase_error_m": 0.0,
        "robot_robot_contact_count": 0,
        "joint_limit_violation": False,
        "torque_limit_violation": False,
        "actuator_saturation": False,
        "physics_steps": 100,
        "pass_delivery_position_m": (1.0, 0.0, 0.115),
        "pass_delivery_error_m": 0.0,
        "pass_delivery_lateral_error_m": 0.0,
        "goalkeeper_enabled": True,
    }
    values.update(overrides)
    return G1SharedWorldResult(**values)  # type: ignore[arg-type]


def _naturalness(**overrides: float) -> G1MotionNaturalnessMetrics:
    values = {
        "contact_joint_acceleration_rms_rad_s2": 90.0,
        "teacher_position_error_rms_rad": 0.4,
        "teacher_velocity_error_rms_rad_s": 3.5,
        "post_contact_joint_acceleration_rms_rad_s2": 45.0,
        "post_contact_root_acceleration_rms_m_s2": 2.5,
        "post_contact_peak_backward_velocity_mps": 0.02,
        "post_contact_support_slip_m": 0.05,
        "torso_roll_peak_rad": 0.2,
        "tail_wobble_index": 0.01,
        "target_error_m": 0.02,
    }
    values.update(overrides)
    return G1MotionNaturalnessMetrics(**values)


def _trajectory() -> dict[str, np.ndarray]:
    return {
        "time": np.asarray((0.0, 0.02)),
        "digest_padding": np.zeros((2, 1), dtype=np.float64),
    }


def test_imitation_candidate_binds_follow_through_and_teacher_blends() -> None:
    candidate = G1ImitationCandidate(0.02, 0.02, 253, 0.09, 0.01, 0.06)
    overrides = candidate.simulation_overrides(Path("prior.json"))

    assert overrides["shooter_motion_prior_velocity_blend"] == 0.02
    assert overrides["shooter_post_policy_forward_velocity_mps"] == 0.06
    assert candidate.candidate_hash.startswith("sha256:")


def test_imitation_trial_accepts_jointly_better_safe_motion() -> None:
    candidate = G1ImitationCandidate(0.02, 0.02, 253, 0.09, 0.01, 0.06)
    parent = _naturalness(
        contact_joint_acceleration_rms_rad_s2=95.0,
        teacher_position_error_rms_rad=0.45,
        teacher_velocity_error_rms_rad_s=4.0,
        post_contact_joint_acceleration_rms_rad_s2=52.0,
        post_contact_root_acceleration_rms_m_s2=3.0,
        post_contact_peak_backward_velocity_mps=0.03,
        post_contact_support_slip_m=0.07,
        torso_roll_peak_rad=0.25,
        target_error_m=0.05,
    )
    trial = evaluate_g1_imitation_trial(
        candidate=candidate,
        result=_result(),
        trajectory=_trajectory(),
        parent_result=_result(target_error_m=0.05),
        parent=parent,
        naturalness=_naturalness(),
    )

    assert trial.eligible
    assert trial.safety_cost == 0.0
    assert not trial.reasons


def test_imitation_trial_rejects_accuracy_forgetting_or_backward_regression() -> None:
    candidate = G1ImitationCandidate(0.02, 0.02, 253, 0.09, 0.01, 0.06)
    parent = _naturalness(target_error_m=0.05)
    regressed = replace(
        parent,
        target_error_m=0.08,
        post_contact_peak_backward_velocity_mps=0.04,
    )
    trial = evaluate_g1_imitation_trial(
        candidate=candidate,
        result=_result(target_error_m=0.08),
        trajectory=_trajectory(),
        parent_result=_result(target_error_m=0.05),
        parent=parent,
        naturalness=regressed,
    )

    assert not trial.eligible
    assert "accuracy_not_forgotten" in trial.reasons
    assert "backward_motion_reduced" in trial.reasons


def test_imitation_trial_rejects_unbound_metric_result_pair() -> None:
    candidate = G1ImitationCandidate(0.02, 0.02, 253, 0.09, 0.01, 0.06)
    with pytest.raises(ValueError, match="candidate naturalness"):
        evaluate_g1_imitation_trial(
            candidate=candidate,
            result=_result(target_error_m=0.02),
            trajectory=_trajectory(),
            parent_result=_result(target_error_m=0.02),
            parent=_naturalness(target_error_m=0.02),
            naturalness=_naturalness(target_error_m=0.03),
        )


def test_imitation_candidate_rejects_unbounded_follow_through() -> None:
    with pytest.raises(ValueError, match="follow-through"):
        G1ImitationCandidate(0.02, 0.02, 253, 0.09, 0.01, 0.20)
