from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw_soccer.media.shot_save_population_video import (
    validate_shot_save_population_video_manifest,
)
from rosclaw_soccer.training.shot_save_population import (
    ShotSavePopulationConfig,
    _metric_specs,
    _metrics,
    validate_shot_save_population_exam,
)

_EVIDENCE = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "shot-save-population-v3/shot-save-population-exam.json"
)
_VIDEO_MANIFEST = _EVIDENCE.with_name("shot-save-population-v3.json")


def _cell(*, save: bool, safe: bool = True, pelvis: float = 0.72) -> dict[str, object]:
    return {
        "safe": safe,
        "challenge_valid": safe,
        "keeper_exam_passed": save and safe,
        "result": {
            "goalkeeper_ball_contact_observed": save,
            "goalkeeper_min_pelvis_height_m": pelvis,
        },
    }


def test_population_suite_is_eight_case_content_bound_and_goal_sized() -> None:
    config = ShotSavePopulationConfig()

    assert len(config.shots) == 8
    assert len({item.policy_hash for item in config.shots}) == 8
    assert max(item.physical_target_y_m for item in config.shots) > 1.20
    assert config.scenario_suite_hash.startswith("sha256:")
    with pytest.raises(ValueError, match="population config"):
        ShotSavePopulationConfig(shots=config.shots[:7])


def test_population_metrics_expose_save_gain_without_hiding_safety() -> None:
    parent = tuple(_cell(save=index != 0) for index in range(8))
    candidate = tuple(_cell(save=True) for _ in range(8))

    assert _metrics(parent)["save_rate"] == pytest.approx(0.875)
    assert _metrics(candidate)["save_rate"] == pytest.approx(1.0)
    assert _metric_specs(ShotSavePopulationConfig())[0].minimum_improvement == pytest.approx(
        0.125
    )
    unsafe = candidate[:-1] + (_cell(save=True, safe=False),)
    assert _metrics(unsafe)["safety_rate"] == pytest.approx(0.875)


@pytest.mark.skipif(not _EVIDENCE.is_file(), reason="S79 population evidence unavailable")
def test_actual_population_evidence_is_current_and_tamper_evident(tmp_path: Path) -> None:
    report = validate_shot_save_population_exam(_EVIDENCE)
    assert report["development_champion_replaced"] is True
    assert report["candidate_metrics"]["save_rate"] == pytest.approx(1.0)
    assert report["promotion_eligible"] is False

    payload = json.loads(_EVIDENCE.read_text(encoding="utf-8"))
    payload["sealed_holdout_used"] = True
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        validate_shot_save_population_exam(tampered)


@pytest.mark.skipif(not _VIDEO_MANIFEST.is_file(), reason="S79 population video unavailable")
def test_actual_population_video_is_bound_and_tamper_evident(tmp_path: Path) -> None:
    manifest = validate_shot_save_population_video_manifest(_VIDEO_MANIFEST)
    assert manifest["visualization_only"] is True
    assert manifest["pixels_used_for_scoring"] is False
    assert manifest["promotion_eligible"] is False

    payload = json.loads(_VIDEO_MANIFEST.read_text(encoding="utf-8"))
    payload["pixels_used_for_scoring"] = True
    tampered = tmp_path / "tampered-video.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        validate_shot_save_population_video_manifest(tampered)
