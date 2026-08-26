from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.skills.team.agility_growth import (
    G1AgilityCandidate,
    G1AgilityMetrics,
    evaluate_g1_agility_neighborhood,
    evaluate_g1_agility_trial,
    measure_g1_agility,
    measure_g1_follow_through_agility,
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
        "goal_crossing_y_m": 0.8,
        "goal_crossing_z_m": 1.5,
        "target_error_m": 0.01,
        "passer_min_pelvis_height_m": 0.75,
        "shooter_min_pelvis_height_m": 0.75,
        "passer_roll_peak_rad": 0.1,
        "passer_pitch_peak_rad": 0.1,
        "shooter_roll_peak_rad": 0.22,
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
    }
    values.update(overrides)
    return G1SharedWorldResult(**values)  # type: ignore[arg-type]


def _naturalness(**overrides: float) -> G1MotionNaturalnessMetrics:
    values = {
        "contact_joint_acceleration_rms_rad_s2": 89.0,
        "teacher_position_error_rms_rad": 0.4,
        "teacher_velocity_error_rms_rad_s": 3.5,
        "post_contact_joint_acceleration_rms_rad_s2": 47.0,
        "post_contact_root_acceleration_rms_m_s2": 2.7,
        "post_contact_peak_backward_velocity_mps": 0.001,
        "post_contact_support_slip_m": 0.055,
        "torso_roll_peak_rad": 0.23,
        "tail_wobble_index": 0.01,
        "target_error_m": 0.01,
    }
    values.update(overrides)
    return G1MotionNaturalnessMetrics(**values)


def _agility(energy: float = 3.0) -> G1AgilityMetrics:
    return G1AgilityMetrics(2.0, 2.5, 0.4, 0.6, energy)


def test_candidate_separates_position_and_velocity_authority() -> None:
    candidate = G1AgilityCandidate(1.15, 1.25)
    overrides = candidate.simulation_overrides(
        motion_prior_path=Path("motion.json"),
        contact_prior_path=Path("contact.json"),
    )

    assert overrides["shooter_motion_prior_joint_scales"] == (1.0,) * 29
    assert overrides["shooter_motion_prior_velocity_joint_scales"] == (
        (1.0,) * 12 + (1.15,) * 3 + (1.25,) * 14
    )
    assert candidate.candidate_hash.startswith("sha256:")


def test_candidate_rejects_unbounded_velocity_authority() -> None:
    with pytest.raises(ValueError, match="waist velocity scale"):
        G1AgilityCandidate(2.1, 1.0)


def test_agility_metric_uses_fixed_contact_window() -> None:
    frames = np.arange(240, 266)
    position = np.zeros((len(frames), 29))
    velocity = np.zeros_like(position)
    position[:, 12:15] = np.linspace(0.0, 0.6, len(frames))[:, None]
    velocity[:, 15:] = 2.0
    metrics = measure_g1_agility(
        {
            "shooter_policy_frame": frames,
            "shooter_joint_position": position,
            "shooter_joint_velocity": velocity,
        }
    )

    assert metrics.waist_excursion_rms_rad > 0.0
    assert metrics.arm_velocity_rms_rad_s == 2.0


def test_follow_through_metric_uses_only_teacher_active_window() -> None:
    position = np.zeros((8, 29))
    velocity = np.zeros_like(position)
    teacher = np.zeros_like(position)
    active = np.asarray((False, False, True, True, True, True, False, False))
    position[active, 12:] = np.arange(4)[:, None] * 0.1
    velocity[active, 12:15] = 1.0
    velocity[active, 15:] = 2.0
    teacher[active, 15:] = 0.05

    metrics = measure_g1_follow_through_agility(
        {
            "shooter_joint_position": position,
            "shooter_joint_velocity": velocity,
            "shooter_agility_prior_velocity_delta": teacher,
            "shooter_agility_prior_target_delta": teacher,
            "shooter_agility_prior_active": active,
        }
    )

    assert metrics.active_frame_count == 4
    assert metrics.waist_velocity_rms_rad_s == 1.0
    assert metrics.arm_velocity_rms_rad_s == 2.0
    assert metrics.arm_excursion_rms_rad == pytest.approx(0.3)
    assert metrics.teacher_velocity_l1_rad_s == pytest.approx(2.8)
    assert metrics.teacher_position_l1_rad == pytest.approx(2.8)


def test_follow_through_metric_reports_inactive_teacher() -> None:
    zeros = np.zeros((4, 29))
    metrics = measure_g1_follow_through_agility(
        {
            "shooter_joint_position": zeros,
            "shooter_joint_velocity": zeros,
            "shooter_agility_prior_velocity_delta": zeros,
            "shooter_agility_prior_active": np.zeros(4, dtype=bool),
        }
    )

    assert metrics.active_frame_count == 0
    assert metrics.upper_body_motion_energy == 0.0


def test_follow_through_metric_supports_parent_counterfactual_window() -> None:
    zeros = np.zeros((25, 29))
    velocity = zeros.copy()
    velocity[:, 15:] = 1.0
    metrics = measure_g1_follow_through_agility(
        {
            "shooter_policy_frame": np.arange(268, 293),
            "shooter_joint_position": zeros,
            "shooter_joint_velocity": velocity,
            "shooter_agility_prior_velocity_delta": zeros,
            "shooter_agility_prior_active": np.zeros(25, dtype=bool),
        },
        center_policy_frame=280,
    )

    assert metrics.active_frame_count == 21
    assert metrics.arm_velocity_rms_rad_s == 1.0
    assert metrics.teacher_velocity_l1_rad_s == 0.0


def test_gate_accepts_bounded_agility_without_forgetting() -> None:
    parent = _naturalness(
        target_error_m=0.019,
        contact_joint_acceleration_rms_rad_s2=90.0,
        post_contact_joint_acceleration_rms_rad_s2=48.0,
        post_contact_root_acceleration_rms_m_s2=2.8,
        post_contact_peak_backward_velocity_mps=0.005,
        post_contact_support_slip_m=0.048,
    )
    trial = evaluate_g1_agility_trial(
        candidate=G1AgilityCandidate(1.15, 1.15),
        result=_result(),
        trajectory={"time": np.asarray((0.0, 0.02))},
        parent_naturalness=parent,
        parent_agility=_agility(3.0),
        naturalness=_naturalness(),
        agility=_agility(3.1),
    )

    assert trial.eligible
    assert not trial.reasons


def test_gate_rejects_visual_motion_that_slips_or_recoils() -> None:
    parent = _naturalness()
    regressed = replace(
        parent,
        post_contact_support_slip_m=0.08,
        post_contact_peak_backward_velocity_mps=0.02,
    )
    trial = evaluate_g1_agility_trial(
        candidate=G1AgilityCandidate(1.5, 1.5),
        result=_result(),
        trajectory={"time": np.asarray((0.0, 0.02))},
        parent_naturalness=parent,
        parent_agility=_agility(),
        naturalness=regressed,
        agility=_agility(4.0),
    )

    assert not trial.eligible
    assert "support_slip_6cm" in trial.reasons
    assert "backward_speed_1cm_s" in trial.reasons


def test_neighborhood_requires_a_safe_local_basin() -> None:
    center = G1AgilityCandidate(1.15, 1.15)
    parent = _naturalness(target_error_m=0.019)

    def trial(candidate: G1AgilityCandidate, slip: float):
        naturalness = _naturalness(post_contact_support_slip_m=slip)
        return evaluate_g1_agility_trial(
            candidate=candidate,
            result=_result(),
            trajectory={"time": np.asarray((0.0, 0.02))},
            parent_naturalness=parent,
            parent_agility=_agility(),
            naturalness=naturalness,
            agility=_agility(3.1),
        )

    neighborhood = evaluate_g1_agility_neighborhood(
        center=center,
        trials=(
            trial(G1AgilityCandidate(1.14, 1.14), 0.05),
            trial(center, 0.05),
            trial(G1AgilityCandidate(1.16, 1.16), 0.07),
        ),
    )

    assert neighborhood.passed
    assert neighborhood.eligible_fraction == pytest.approx(2 / 3)
