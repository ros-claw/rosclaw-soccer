from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.skills.team.composite_imitation import (
    G1CompositeImitationCandidate,
    G1ContactImitationMetrics,
    evaluate_g1_composite_imitation_trial,
)
from rosclaw_soccer.skills.team.imitation_learning import G1MotionNaturalnessMetrics
from rosclaw_soccer.skills.team.shared_world import G1SharedWorldResult


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
        "post_contact_peak_backward_velocity_mps": 0.005,
        "post_contact_support_slip_m": 0.04,
        "torso_roll_peak_rad": 0.2,
        "tail_wobble_index": 0.01,
        "target_error_m": 0.02,
    }
    values.update(overrides)
    return G1MotionNaturalnessMetrics(**values)


def _trajectory() -> dict[str, np.ndarray]:
    return {"time": np.asarray((0.0, 0.02)), "padding": np.zeros((2, 1))}


def test_composite_candidate_binds_both_teacher_paths() -> None:
    candidate = G1CompositeImitationCandidate(0.0025, 253, 0.09, 0.06)
    overrides = candidate.simulation_overrides(
        motion_prior_path=Path("motion.json"),
        contact_prior_path=Path("contact.json"),
    )

    assert overrides["shooter_motion_prior_path"] == Path("motion.json")
    assert overrides["shooter_contact_prior_path"] == Path("contact.json")
    assert overrides["shooter_contact_prior_position_blend"] == 0.0025
    assert candidate.candidate_hash.startswith("sha256:")


def test_composite_gate_accepts_safe_plasticity_without_forgetting() -> None:
    candidate = G1CompositeImitationCandidate(0.0025, 253, 0.09, 0.06)
    parent = _naturalness(
        contact_joint_acceleration_rms_rad_s2=95.0,
        post_contact_joint_acceleration_rms_rad_s2=50.0,
        post_contact_support_slip_m=0.06,
        torso_roll_peak_rad=0.25,
    )
    trial = evaluate_g1_composite_imitation_trial(
        candidate=candidate,
        result=_result(),
        trajectory=_trajectory(),
        parent_result=_result(),
        parent=parent,
        parent_contact=G1ContactImitationMetrics(0.70, 0.0, 0.0),
        naturalness=_naturalness(),
        contact=G1ContactImitationMetrics(0.69, 0.001, 0.01),
    )

    assert trial.eligible
    assert trial.safety_cost == 0.0
    assert not trial.reasons


def test_composite_gate_rejects_teacher_gain_that_destabilizes_support() -> None:
    candidate = G1CompositeImitationCandidate(0.0025, 253, 0.09, 0.06)
    parent = _naturalness(post_contact_support_slip_m=0.06, torso_roll_peak_rad=0.25)
    regressed = replace(parent, post_contact_support_slip_m=0.09, torso_roll_peak_rad=0.30)
    trial = evaluate_g1_composite_imitation_trial(
        candidate=candidate,
        result=_result(),
        trajectory=_trajectory(),
        parent_result=_result(),
        parent=parent,
        parent_contact=G1ContactImitationMetrics(0.70, 0.0, 0.0),
        naturalness=regressed,
        contact=G1ContactImitationMetrics(0.60, 0.002, 0.01),
    )

    assert not trial.eligible
    assert "support_slip_reduced" in trial.reasons
    assert "roll_reduced" in trial.reasons


def test_composite_candidate_rejects_unbounded_or_empty_joint_authority() -> None:
    with pytest.raises(ValueError, match="at least one joint"):
        G1CompositeImitationCandidate(
            0.0025,
            253,
            0.09,
            0.06,
            contact_joint_scales=(0.0,) * 6,
        )
    with pytest.raises(ValueError, match="contact imitation blend"):
        G1CompositeImitationCandidate(0.1, 253, 0.09, 0.06)
