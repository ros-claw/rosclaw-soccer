from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.media.continuous_second_striker_save_video import (
    validate_continuous_second_striker_save_video_manifest,
)
from rosclaw_soccer.skills.team.shared_world import G1SharedWorldResult
from rosclaw_soccer.training.continuous_second_striker_save_exam import (
    ContinuousSecondStrikerSaveExamConfig,
    _continuous_metrics,
    validate_continuous_second_striker_save_exam,
)

_EVIDENCE = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s109-continuous-second-striker-save-v1/evidence.json"
)
_VIDEO_MANIFEST = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s109-continuous-second-striker-showcase-v2/s109-four-g1-two-save.json"
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
        second_striker_contact_time_sec=16.95,
        goalkeeper_second_glove_contact_time_sec=17.51,
    )


def test_physical_second_striker_contract_is_bounded_and_role_specific() -> None:
    config = ContinuousSecondStrikerSaveExamConfig()
    striker = config.striker
    assert config.activation_ceiling == "SIM_ONLY"
    assert not config.hardware_authorized
    assert striker.goalkeeper_punch_force_n == pytest.approx(90.0)
    assert striker.goalkeeper_punch_outward_force_scale == pytest.approx(0.0)
    assert striker.foot_pitch_offset == pytest.approx(0.10)
    assert striker.loft_synergy == pytest.approx(0.10)
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="punch force"):
        replace(striker, goalkeeper_punch_force_n=121.0)
    with pytest.raises(ValueError, match="strike pocket"):
        replace(striker, ball_origin_m=(3.0, 0.282, 0.115))


def test_continuous_metrics_bind_actor_memories_and_outward_deflection() -> None:
    time = np.arange(1001, dtype=np.float64) * 0.02
    velocity = np.zeros((time.size, 6), dtype=np.float64)
    velocity[848:876, 0] = 8.5
    velocity[876:880, :3] = (-1.0, -4.6, 2.5)
    actor_active = np.zeros(time.size, dtype=np.bool_)
    actor_active[846:848] = True
    target_active = np.zeros(time.size, dtype=np.bool_)
    target_active[840:853] = True
    torque_active = np.zeros(time.size, dtype=np.bool_)
    torque_active[844:852] = True
    actor_torque = np.zeros((time.size, 29), dtype=np.float64)
    actor_torque[846:848, 6] = 61.0
    observed = np.zeros(time.size, dtype=np.bool_)
    observed[848:879] = True
    flight_start = np.full(time.size, np.nan, dtype=np.float64)
    flight_start[848:879] = 16.96
    reaction = np.zeros(time.size, dtype=np.bool_)
    reaction[850:879] = True
    trajectory = {
        "time": time,
        "second_ball_velocity": velocity,
        "second_striker_ballistic_actor_active": actor_active,
        "second_striker_ballistic_actor_torque": actor_torque,
        "second_striker_ballistic_contact_active": target_active,
        "second_striker_ballistic_contact_torque_active": torque_active,
        "goalkeeper_observed_flight_active": observed,
        "goalkeeper_observed_flight_start_sec": flight_start,
        "goalkeeper_reaction_active": reaction,
    }
    metrics = _continuous_metrics(_result(), trajectory)
    assert metrics["valid"] is True
    assert metrics["actor_active_frame_count"] == 2
    assert metrics["actor_peak_torque_nm"] == pytest.approx(61.0)
    assert metrics["contact_target_memory_active_frame_count"] == 13
    assert metrics["upper_corner_torque_memory_active_frame_count"] == 8
    assert metrics["new_flight_start_time_sec"] == pytest.approx(16.96)
    assert metrics["post_glove_minimum_forward_speed_mps"] == pytest.approx(-1.0)
    assert metrics["post_glove_peak_outward_speed_mps"] == pytest.approx(4.6)


def test_external_continuous_second_striker_evidence_if_available() -> None:
    if not _EVIDENCE.is_file():
        pytest.skip("external S109 physics evidence is not present")
    payload = validate_continuous_second_striker_save_exam(_EVIDENCE)
    assert payload["passed"] is True
    assert payload["ball_cannon_used"] is False
    assert payload["reset_or_teleport_used"] is False


def test_external_continuous_second_striker_video_if_available() -> None:
    if not _VIDEO_MANIFEST.is_file():
        pytest.skip("external S109 video manifest is not present")
    payload = validate_continuous_second_striker_save_video_manifest(_VIDEO_MANIFEST)
    assert payload["four_g1_visible"] is True
    assert payload["two_physical_balls_visible"] is True
    assert payload["ball_cannon_used"] is False
    assert payload["pixels_used_for_scoring"] is False
