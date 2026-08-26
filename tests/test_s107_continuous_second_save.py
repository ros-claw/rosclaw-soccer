from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.skills.goalkeeper_v2.observations import (
    GoalkeeperActorObservation,
    GoalkeeperActorObserver,
)
from rosclaw_soccer.skills.team.shared_world import (
    G1SecondThreatConfig,
    G1SharedWorldResult,
)
from rosclaw_soccer.training.continuous_second_save_exam import (
    ContinuousSecondSaveExamConfig,
    _continuity_metrics,
    validate_continuous_second_save_exam,
)

_EVIDENCE = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s107-continuous-second-save-exam-v1/evidence.json"
)


def _observe(
    observer: GoalkeeperActorObserver, timestamp: float, ball_x: float
) -> GoalkeeperActorObservation:
    return observer.observe(
        timestamp_sec=timestamp,
        ball_relative_position_m=np.asarray((ball_x, 0.0, 0.3), dtype=np.float64),
        gravity_orientation=np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
        root_linear_velocity_mps=np.zeros(3, dtype=np.float64),
        angular_velocity_rad_s=np.zeros(3, dtype=np.float64),
        joint_position_rad=np.zeros(29, dtype=np.float64),
        joint_velocity_rad_s=np.zeros(29, dtype=np.float64),
        previous_action_rad=np.zeros(29, dtype=np.float64),
    )


def _result() -> G1SharedWorldResult:
    return G1SharedWorldResult(
        finite_state=True,
        pass_contact_observed=True,
        shot_contact_observed=True,
        pass_contact_time_sec=1.0,
        shot_contact_time_sec=5.0,
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
        physics_steps=10_000,
        second_threat_rearmed=True,
        second_threat_rearm_time_sec=16.4,
        second_threat_launch_observed=True,
        second_threat_launch_time_sec=17.0,
        goalkeeper_second_glove_contact_observed=True,
        goalkeeper_second_glove_contact_time_sec=17.85,
    )


def test_second_threat_contract_is_bounded_sim_only() -> None:
    threat = G1SecondThreatConfig()
    exam = ContinuousSecondSaveExamConfig()
    assert threat.activation_ceiling == "SIM_ONLY"
    assert not threat.hardware_authorized
    assert threat.force_duration_sec == pytest.approx(0.08)
    assert threat.goalkeeper_punch_force_n <= 120.0
    assert exam.activation_ceiling == "SIM_ONLY"
    assert not exam.hardware_authorized
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(threat, hardware_authorized=True)
    with pytest.raises(ValueError, match="force"):
        replace(threat, maximum_force_n=121.0)
    with pytest.raises(ValueError, match="punch"):
        replace(threat, goalkeeper_punch_force_n=121.0)
    with pytest.raises(ValueError, match="lanes"):
        replace(exam, lane_ids=("left-inner", "left-inner"))
    with pytest.raises(ValueError, match="non-commercial SIM_ONLY"):
        replace(exam, commercial_use_allowed=True)


def test_goalkeeper_observer_rearm_starts_a_new_causal_epoch() -> None:
    observer = GoalkeeperActorObserver(flight_velocity_threshold_mps=0.10)
    _observe(observer, 0.0, 3.0)
    first = _observe(observer, 0.02, 2.98)
    assert first.observed_flight_start_sec == pytest.approx(0.02)

    observer.rearm()
    reset = _observe(observer, 1.0, 2.0)
    second = _observe(observer, 1.02, 1.98)
    assert reset.observed_flight_start_sec is None
    assert not reset.ball_history_ready
    assert second.observed_flight_start_sec == pytest.approx(1.02)


def test_continuity_metrics_detect_real_force_and_no_state_jump() -> None:
    time = np.arange(1001, dtype=np.float64) * 0.02
    ball = np.zeros((time.size, 7), dtype=np.float64)
    ball[:, 0] = np.linspace(2.0, 3.0, time.size)
    ball[:, 2] = 0.115
    velocity = np.zeros((time.size, 6), dtype=np.float64)
    velocity[:, 0] = 0.05
    velocity[850:860, 0] = np.linspace(0.05, 6.0, 10)
    force = np.zeros((time.size, 3), dtype=np.float64)
    force[850:854, 0] = 60.0
    observed = np.zeros(time.size, dtype=np.bool_)
    observed[854:893] = True
    flight_start = np.full(time.size, np.nan, dtype=np.float64)
    flight_start[854:893] = 17.08
    reaction = np.zeros(time.size, dtype=np.bool_)
    reaction[856:893] = True
    epochs = np.ones(time.size, dtype=np.int64)
    epochs[893:] = 2
    trajectory = {
        "time": time,
        "ball_pose": ball,
        "ball_velocity": velocity,
        "goalkeeper_pelvis_pose": np.zeros((time.size, 7), dtype=np.float64),
        "goalkeeper_joint_position": np.zeros((time.size, 29), dtype=np.float64),
        "second_threat_launcher_force": force,
        "goalkeeper_observed_flight_active": observed,
        "goalkeeper_observed_flight_start_sec": flight_start,
        "goalkeeper_reaction_active": reaction,
        "goalkeeper_contact_epoch": epochs,
    }
    metrics = _continuity_metrics(result=_result(), trajectory=trajectory)
    assert metrics["valid"] is True
    assert metrics["launcher_active_frame_count"] == 4
    assert metrics["launcher_telemetry_peak_force_n"] == pytest.approx(60.0)
    assert metrics["post_launch_speed_gain_mps"] > 4.0
    assert metrics["flight_epoch_clear_at_launch"] is True
    assert metrics["new_flight_epoch_observed"] is True
    assert metrics["new_flight_start_time_sec"] == pytest.approx(17.08)
    assert metrics["causal_reaction_observed"] is True
    assert metrics["maximum_contact_epoch"] == 2

    discontinuous = dict(trajectory)
    jumped = ball.copy()
    jumped[820, 0] += 1.0
    discontinuous["ball_pose"] = jumped
    bad = _continuity_metrics(result=_result(), trajectory=discontinuous)
    assert bad["rearm_ball_position_step_m"] > 0.9


def test_external_continuous_second_save_evidence_if_available() -> None:
    if not _EVIDENCE.is_file():
        pytest.skip("external S107 physics evidence is not present")
    payload = validate_continuous_second_save_exam(_EVIDENCE)
    assert payload["passed"] is True
    assert payload["second_striker_claimed"] is False
