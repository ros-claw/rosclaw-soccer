from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.media.dive_athlete_video import (
    validate_dive_athlete_video_manifest,
)
from rosclaw_soccer.training.dive_athlete_cpu_exam import (
    validate_dive_athlete_cpu_exam_report,
)
from rosclaw_soccer.training.dive_athlete_expert import (
    DiveAthleteExpertConfig,
    build_physics_margin_dive_teacher,
    dive_athlete_features_numpy,
)
from rosclaw_soccer.training.goalkeeper_dive_option import (
    GoalkeeperBalancedDiveSeed,
    GoalkeeperDiveDirection,
    mirror_g1_joint_positions,
)


def _seed() -> GoalkeeperBalancedDiveSeed:
    rng = np.random.default_rng(102)
    left = rng.uniform(-0.2, 0.2, size=(3, 29))
    joint = np.stack((left, mirror_g1_joint_positions(left)))
    return GoalkeeperBalancedDiveSeed(
        source_atlas_hash="sha256:" + "a" * 64,
        source_direction=GoalkeeperDiveDirection.LEFT,
        source_start_frame=10,
        source_end_frame=12,
        joint_position_rad=joint,
        root_displacement_m=np.zeros((2, 3, 3), dtype=np.float64),
        frame_rate_hz=50.0,
        source_lateral_displacement_m=0.4,
    )


def test_dive_athlete_contract_is_noncommercial_sim_only() -> None:
    config = DiveAthleteExpertConfig()
    assert config.config_hash.startswith("sha256:")
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="ankle-roll"):
        replace(config, inward_ankle_roll_teacher_rad=0.25)


def test_physics_margin_teacher_is_exactly_bilateral_and_only_changes_ankles() -> None:
    seed = _seed()
    teacher = build_physics_margin_dive_teacher(seed, inward_ankle_roll_rad=0.2)

    assert np.allclose(teacher[1], mirror_g1_joint_positions(teacher[0]))
    assert np.all(teacher[:, :, 5] == 0.2)
    assert np.all(teacher[:, :, 11] == -0.2)
    retained = np.ones(29, dtype=np.bool_)
    retained[[5, 11]] = False
    assert np.array_equal(teacher[:, :, retained], seed.joint_position_rad[:, :, retained])
    assert not teacher.flags.writeable


def test_dive_athlete_features_fail_closed_on_nonfinite_context() -> None:
    values = np.asarray((0.0, 0.5, 1.0), dtype=np.float64)
    features = dive_athlete_features_numpy(
        phase=values,
        target_lateral_m=np.full(3, 0.4),
        target_height_m=np.full(3, 1.45),
        duration_sec=np.full(3, 1.16),
        contact_phase=np.full(3, 0.32),
    )
    assert features.shape == (3, 10)
    with pytest.raises(ValueError, match="finite"):
        dive_athlete_features_numpy(
            phase=np.asarray((0.0, np.nan, 1.0)),
            target_lateral_m=np.full(3, 0.4),
            target_height_m=np.full(3, 1.45),
            duration_sec=np.full(3, 1.16),
            contact_phase=np.full(3, 0.32),
        )


def test_current_dive_athlete_cpu_exam_is_content_bound_when_available() -> None:
    evidence = Path(
        "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
        "s102-dive-athlete-cpu-exam-v1/evidence.json"
    )
    if not evidence.is_file():
        pytest.skip("S102 external CPU MuJoCo evidence is unavailable")
    report = validate_dive_athlete_cpu_exam_report(evidence)
    assert report["candidate_repaired_rejected_teacher"] is True
    assert report["metrics"]["bilateral_symmetry_error_rad"] == 0.0
    assert report["metrics"]["minimum_candidate_lateral_displacement_m"] > 0.28


def test_current_dive_athlete_video_is_content_bound_when_available() -> None:
    manifest = Path(
        "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
        "s102-dive-athlete-showcase-v1/s102-neural-dive-athlete.json"
    )
    if not manifest.is_file():
        pytest.skip("S102 external video evidence is unavailable")

    report = validate_dive_athlete_video_manifest(manifest)

    assert report["case_count"] == 4
    assert report["pixels_used_for_scoring"] is False
    assert report["promotion_eligible"] is False
