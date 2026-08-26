from __future__ import annotations

from rosclaw_soccer.training.goalkeeper_curriculum import (
    GoalkeeperCurriculumEvidence,
    curriculum_manifest,
    decide_curriculum_stage,
    default_goalkeeper_curriculum,
)


def test_default_curriculum_progresses_to_sequential_saves() -> None:
    stages = default_goalkeeper_curriculum()
    assert tuple(stage.name for stage in stages) == (
        "stand_reach",
        "shuffle_save",
        "dive_land_recover",
        "sequential_saves",
    )
    assert stages[-1].second_shot_probability > 0.0
    assert stages[-1].minimum_second_save_rate > 0.0
    assert stages[-1].shot_deadline_range_sec[0] == 0.40
    assert len({stage.stage_hash for stage in stages}) == len(stages)


def test_curriculum_never_advances_from_training_reward_alone() -> None:
    stage = default_goalkeeper_curriculum()[0]
    decision = decide_curriculum_stage(
        stage,
        GoalkeeperCurriculumEvidence(
            completed_updates=stage.minimum_updates,
            first_save_rate=1.0,
            recovery_rate=1.0,
            second_save_rate=1.0,
            unsafe_rate=0.0,
            strict_replay=False,
        ),
    )
    assert decision.action == "HOLD"
    assert "strict_replay_missing" in decision.reasons


def test_curriculum_fails_closed_on_any_unsafe_episode() -> None:
    stage = default_goalkeeper_curriculum()[0]
    decision = decide_curriculum_stage(
        stage,
        GoalkeeperCurriculumEvidence(
            completed_updates=stage.maximum_updates,
            first_save_rate=1.0,
            recovery_rate=1.0,
            second_save_rate=1.0,
            unsafe_rate=0.001,
            strict_replay=True,
        ),
    )
    assert decision.action == "REJECT_UNSAFE"


def test_curriculum_manifest_has_no_promotion_authority() -> None:
    manifest = curriculum_manifest()
    assert manifest["curriculum_hash"].startswith("sha256:")
    assert manifest["activation_ceiling"] == "SIM_ONLY"
    assert not manifest["promotion_authority"]
