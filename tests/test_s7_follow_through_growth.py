from __future__ import annotations

from dataclasses import replace

import numpy as np

from rosclaw_soccer.skills.team.agility_growth import G1FollowThroughAgilityMetrics
from rosclaw_soccer.skills.team.follow_through_growth import (
    G1FollowThroughCandidate,
    default_g1_follow_through_candidates,
    evaluate_g1_follow_through_trial,
)
from rosclaw_soccer.skills.team.imitation_learning import G1MotionNaturalnessMetrics
from rosclaw_soccer.skills.team.shared_world import G1SharedWorldResult


def _result() -> G1SharedWorldResult:
    return G1SharedWorldResult(
        finite_state=True,
        pass_contact_observed=True,
        shot_contact_observed=True,
        pass_contact_time_sec=1.0,
        shot_contact_time_sec=2.0,
        pass_peak_ball_speed_mps=1.0,
        shot_peak_ball_speed_mps=8.0,
        goal_crossed=True,
        goal_plane_crossed=True,
        goal_crossing_y_m=0.8,
        goal_crossing_z_m=1.5,
        target_error_m=0.01,
        passer_min_pelvis_height_m=0.75,
        shooter_min_pelvis_height_m=0.75,
        passer_roll_peak_rad=0.1,
        passer_pitch_peak_rad=0.1,
        shooter_roll_peak_rad=0.2,
        shooter_pitch_peak_rad=0.2,
        passer_tail_wobble_index=0.01,
        shooter_tail_wobble_index=0.01,
        receiver_phase_hold_frames=0,
        receiver_phase_advance_frames=0,
        receiver_max_ball_phase_error_m=0.0,
        robot_robot_contact_count=0,
        joint_limit_violation=False,
        torque_limit_violation=False,
        actuator_saturation=False,
        physics_steps=100,
        pass_delivery_position_m=(1.0, 0.0, 0.115),
        pass_delivery_error_m=0.0,
        pass_delivery_lateral_error_m=0.0,
    )


def _naturalness() -> G1MotionNaturalnessMetrics:
    return G1MotionNaturalnessMetrics(89.0, 0.4, 3.5, 47.0, 2.7, 0.002, 0.047, 0.22, 0.01, 0.01)


def _agility(*, excursion: float, energy: float, teacher: float = 0.0):
    return G1FollowThroughAgilityMetrics(21, 0.5, 0.6, 0.08, excursion, energy, teacher, 0.0)


def test_follow_through_candidate_is_arm_only_and_bounded() -> None:
    candidate = G1FollowThroughCandidate(0.5, 288)
    overrides = candidate.simulation_overrides(mosaic_prior_path="prior.json")  # type: ignore[arg-type]

    assert overrides["shooter_agility_prior_joint_scales"] == (0.0,) * 15 + (1.0,) * 14
    assert len(default_g1_follow_through_candidates()) == 9


def test_follow_through_gate_requires_visible_gain() -> None:
    parent = _agility(excursion=0.10, energy=0.10)
    trial = evaluate_g1_follow_through_trial(
        candidate=G1FollowThroughCandidate(0.5, 288),
        result=_result(),
        trajectory={"time": np.asarray((0.0, 0.02))},
        parent_naturalness=_naturalness(),
        parent_agility=parent,
        naturalness=_naturalness(),
        agility=_agility(excursion=0.12, energy=0.12, teacher=1.0),
    )

    assert trial.eligible


def test_follow_through_gate_rejects_motion_without_retention() -> None:
    parent_naturalness = _naturalness()
    trial = evaluate_g1_follow_through_trial(
        candidate=G1FollowThroughCandidate(0.5, 288),
        result=_result(),
        trajectory={"time": np.asarray((0.0, 0.02))},
        parent_naturalness=parent_naturalness,
        parent_agility=_agility(excursion=0.10, energy=0.10),
        naturalness=replace(parent_naturalness, post_contact_support_slip_m=0.08),
        agility=_agility(excursion=0.15, energy=0.20, teacher=1.0),
    )

    assert not trial.eligible
    assert "support_slip_6cm" in trial.reasons
