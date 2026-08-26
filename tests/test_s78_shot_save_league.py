from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.media.shot_save_league_video import (
    validate_shot_save_league_video_manifest,
)
from rosclaw_soccer.skills.team.shared_world import G1SharedWorldResult
from rosclaw_soccer.training.shot_save_league import (
    ShotSaveGoalkeeperPolicy,
    ShotSaveLeagueConfig,
    ShotSaveStrikerPolicy,
    _cell,
    _simulation_kwargs,
    validate_shot_save_growth_round,
)

_EVIDENCE = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "shot-save-league-v6/shot-save-growth-round.json"
)
_VIDEO_MANIFEST = _EVIDENCE.with_name("shot-save-growth-v6.json")


def _result(**overrides: object) -> G1SharedWorldResult:
    values: dict[str, object] = {
        "finite_state": True,
        "pass_contact_observed": True,
        "shot_contact_observed": True,
        "pass_contact_time_sec": 5.0,
        "shot_contact_time_sec": 7.0,
        "pass_peak_ball_speed_mps": 1.5,
        "shot_peak_ball_speed_mps": 9.0,
        "goal_crossed": True,
        "goal_plane_crossed": True,
        "goal_crossing_y_m": 0.8,
        "goal_crossing_z_m": 0.2,
        "target_error_m": 0.01,
        "passer_min_pelvis_height_m": 0.68,
        "shooter_min_pelvis_height_m": 0.67,
        "passer_roll_peak_rad": 0.2,
        "passer_pitch_peak_rad": 0.2,
        "shooter_roll_peak_rad": 0.2,
        "shooter_pitch_peak_rad": 0.2,
        "passer_tail_wobble_index": 0.0,
        "shooter_tail_wobble_index": 0.0,
        "receiver_phase_hold_frames": 0,
        "receiver_phase_advance_frames": 0,
        "receiver_max_ball_phase_error_m": 0.0,
        "robot_robot_contact_count": 0,
        "joint_limit_violation": False,
        "torque_limit_violation": False,
        "actuator_saturation": False,
        "physics_steps": 100,
        "goalkeeper_enabled": True,
        "goalkeeper_min_pelvis_height_m": 0.72,
        "goalkeeper_ball_contact_observed": False,
        "goalkeeper_ball_contact_time_sec": None,
        "goalkeeper_save_observed": False,
    }
    values.update(overrides)
    return G1SharedWorldResult(**values)  # type: ignore[arg-type]


def _trajectory() -> dict[str, np.ndarray]:
    return {"time": np.arange(4, dtype=np.float64)}


def test_league_cell_scores_role_specific_best_responses() -> None:
    config = ShotSaveLeagueConfig()
    attack = _cell(
        stage="STRIKER_BEST_RESPONSE",
        striker=config.striker_candidates[0],
        goalkeeper=config.parent_goalkeeper,
        result=_result(),
        trajectory=_trajectory(),
        config=config,
    )
    defense = _cell(
        stage="GOALKEEPER_BEST_RESPONSE",
        striker=config.striker_candidates[0],
        goalkeeper=config.goalkeeper_candidates[0],
        result=_result(
            goal_crossed=False,
            goal_plane_crossed=False,
            goal_crossing_y_m=None,
            goal_crossing_z_m=None,
            target_error_m=None,
            goalkeeper_ball_contact_observed=True,
            goalkeeper_ball_contact_time_sec=8.0,
            goalkeeper_save_observed=True,
        ),
        trajectory=_trajectory(),
        config=config,
    )

    assert attack["striker_eligible"] and not attack["goalkeeper_eligible"]
    assert defense["goalkeeper_eligible"] and not defense["striker_eligible"]


def test_league_policies_reject_unsafe_or_out_of_contract_values() -> None:
    with pytest.raises(ValueError, match="striker policy"):
        ShotSaveStrikerPolicy("bad", 1.386, 0.2, 0.0, 0.2)
    with pytest.raises(ValueError, match="goalkeeper policy"):
        ShotSaveGoalkeeperPolicy("bad", 0.20, 0.1)


def test_new_striker_options_preserve_the_parent_approach_posture() -> None:
    config = ShotSaveLeagueConfig()
    kwargs = _simulation_kwargs(
        striker=config.striker_candidates[1],
        goalkeeper=config.parent_goalkeeper,
    )

    assert kwargs["shooter_parameter_overrides"]["pelvis_yaw_offset"] == pytest.approx(0.175)


@pytest.mark.skipif(not _EVIDENCE.is_file(), reason="S78 evidence unavailable")
def test_actual_league_evidence_is_bound_and_tamper_evident(tmp_path: Path) -> None:
    report = validate_shot_save_growth_round(_EVIDENCE)
    assert report["attacker_best_response_succeeded"] is True
    assert report["defender_best_response_succeeded"] is True
    assert report["promotion_eligible"] is False

    tampered = tmp_path / "tampered.json"
    payload = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    payload["sealed_holdout_used"] = True
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        validate_shot_save_growth_round(tampered)


@pytest.mark.skipif(not _VIDEO_MANIFEST.is_file(), reason="S78 video unavailable")
def test_actual_league_video_is_bound_and_tamper_evident(tmp_path: Path) -> None:
    manifest = validate_shot_save_league_video_manifest(_VIDEO_MANIFEST)
    assert manifest["visualization_only"] is True
    assert manifest["pixels_used_for_scoring"] is False

    tampered = tmp_path / "tampered-video.json"
    payload = json.loads(_VIDEO_MANIFEST.read_text(encoding="utf-8"))
    payload["promotion_eligible"] = True
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        validate_shot_save_league_video_manifest(tampered)
