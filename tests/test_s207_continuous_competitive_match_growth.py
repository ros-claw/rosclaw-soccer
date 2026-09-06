from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.growth.competitive_match_assessment import CompetitiveMatchThresholds
from rosclaw_soccer.media.continuous_competitive_match_video import (
    _first_touch_time,
    _timeline,
)
from rosclaw_soccer.training.continuous_competitive_match_growth import (
    default_continuous_match_config,
    default_continuous_match_scenario,
    run_continuous_competitive_match_growth,
    validate_continuous_competitive_match_growth,
)


def test_continuous_match_defaults_define_a_hard_sim_only_exam() -> None:
    world = default_continuous_match_config()
    scenario = default_continuous_match_scenario()
    thresholds = CompetitiveMatchThresholds()

    assert world.simulation_duration_sec == 10.0
    assert world.maximum_speed_mps == 0.70
    assert world.minimum_player_separation_m == 1.10
    assert world.activation_ceiling == "SIM_ONLY"
    assert not world.hardware_authorized
    assert scenario.ball_initial_position_m == (1.92, -0.80, 0.115)
    assert scenario.ball_initial_velocity_mps == (0.0, 0.0, 0.0)
    assert thresholds.minimum_shot_speed_mps == 3.0
    assert thresholds.minimum_shot_goalward_speed_mps == 1.5
    assert thresholds.minimum_save_velocity_change_mps == 0.50


def test_continuous_growth_refuses_to_overwrite_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "operator-data.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="must be empty"):
        run_continuous_competitive_match_growth(
            evidence_dir=evidence,
            asset_root=tmp_path / "missing-assets",
        )

    assert (evidence / "operator-data.txt").read_text(encoding="utf-8") == "preserve"


def test_continuous_validator_fails_closed_on_an_unbound_report(
    tmp_path: Path,
) -> None:
    path = tmp_path / "report.json"
    path.write_text('{"passed": false}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="trajectory binding is absent"):
        validate_continuous_competitive_match_growth(path)


def test_video_timeline_binds_first_touch_and_has_three_views() -> None:
    count = 180
    time = 0.02 * (np.arange(count, dtype=np.float64) + 1.0)
    trajectory: dict[str, np.ndarray] = {
        "time": time,
        "ball_contact_agent_code": np.zeros(count, dtype=np.int64),
        "ball_contact_effector_code": np.zeros(count, dtype=np.int64),
        "ball_pose": np.tile(np.asarray((0.0, 0.0, 0.115, 1.0, 0.0, 0.0, 0.0)), (count, 1)),
        "blue_finisher_pelvis_pose": np.zeros((count, 7), dtype=np.float64),
        "red_finisher_pelvis_pose": np.zeros((count, 7), dtype=np.float64),
    }
    trajectory["ball_contact_agent_code"][100] = 2
    trajectory["ball_contact_effector_code"][100] = 1
    events = [{"skill": "pass", "time_sec": 1.0, "target_agent_id": "red.finisher"}]

    assert _first_touch_time(
        trajectory,
        events,
        agent_ids=("blue.finisher", "red.finisher"),
    ) == pytest.approx(time[100])
    clips = _timeline(trajectory, fps=30)
    assert {frame.camera for clip in clips for frame in clip.frames} == {
        "broadcast",
        "counter",
        "touchline",
        "wide",
    }
    assert sum(len(clip.frames) for clip in clips) > 10 * 30
