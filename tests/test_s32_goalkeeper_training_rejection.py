from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from rosclaw_soccer.training.goalkeeper_physics_ppo import (
    GoalkeeperPhysicsPPOConfig,
    _candidate_is_selectable,
    _candidate_truth_key,
    _exploration_truth_key,
    _height_stratum_event_counts,
    _held_out_selection_seed,
    _safe_continuation_truth_key,
    _selection_outcome,
)


def test_deterministic_selection_seed_is_fixed_across_candidate_iterations() -> None:
    base = _held_out_selection_seed(random_seed=12301, rank=0)
    assert base == 712302
    assert _held_out_selection_seed(random_seed=12301, rank=3) == base + 3
    with pytest.raises(ValueError, match="non-negative"):
        _held_out_selection_seed(random_seed=12301, rank=-1)
    panel = GoalkeeperPhysicsPPOConfig(
        deterministic_selection_seed_count=3,
        deterministic_selection_seed_stride=1009,
    )
    assert panel.deterministic_selection_seed_count == 3
    with pytest.raises(ValueError, match="selection seed count"):
        GoalkeeperPhysicsPPOConfig(deterministic_selection_seed_count=9)


def _candidate(**overrides: float) -> dict[str, float]:
    candidate = {
        "first_save_rate": 0.25,
        "minimum_first_save_stratum_rate": 0.25,
        "first_hand_save_rate": 0.10,
        "second_attempt_save_rate": 0.20,
        "second_attempt_hand_save_rate": 0.05,
        "second_save_rate": 0.10,
        "second_hand_save_rate": 0.05,
        "anatomical_selection_score": 12.0,
        "failed_rate": 0.0,
        "quarantined_rate": 0.0,
        "nonfinite_quarantine_rate": 0.0,
        "maximum_root_angular_speed_rad_s": 3.0,
        "mean_maximum_root_angular_speed_rad_s": 2.0,
        "mean_maximum_hand_displacement_m": 0.30,
    }
    candidate.update(overrides)
    return candidate


def test_no_safe_rollout_is_a_journalable_rejection() -> None:
    outcome = _selection_outcome(
        checkpoint=None,
        best_reward=float("-inf"),
        best_selection_score=float("-inf"),
        best_rollout_iteration=None,
    )

    assert outcome["promotion_status"] == "REJECTED_NO_SAFE_CANDIDATE"
    assert outcome["candidate_available"] is False
    assert outcome["candidate_checkpoint"] is None
    assert outcome["candidate_checkpoint_bytes"] == 0
    assert outcome["best_mean_episode_reward"] is None
    assert outcome["best_anatomical_selection_score"] is None
    assert outcome["rejection_reasons"]
    assert "Infinity" not in json.dumps(outcome, sort_keys=True)


def test_safe_continuation_prefers_weakest_stratum_and_rejects_failures() -> None:
    balanced = _candidate(
        first_save_rate=0.20,
        minimum_first_save_stratum_rate=0.15,
        maximum_root_angular_speed_rad_s=4.0,
    )
    lopsided = _candidate(
        first_save_rate=0.40,
        minimum_first_save_stratum_rate=0.05,
        maximum_root_angular_speed_rad_s=3.0,
    )
    assert _safe_continuation_truth_key(
        balanced, maximum_root_angular_speed_rad_s=5.0
    ) > _safe_continuation_truth_key(lopsided, maximum_root_angular_speed_rad_s=5.0)
    assert (
        _safe_continuation_truth_key(
            _candidate(failed_rate=0.01), maximum_root_angular_speed_rad_s=3.5
        )
        is None
    )
    assert (
        _safe_continuation_truth_key(
            _candidate(quarantined_rate=0.01), maximum_root_angular_speed_rad_s=3.5
        )
        is None
    )
    assert (
        _safe_continuation_truth_key(
            _candidate(maximum_root_angular_speed_rad_s=3.51),
            maximum_root_angular_speed_rad_s=3.5,
        )
        is None
    )


def test_selected_rollout_is_bound_to_the_written_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"content-bound-candidate")

    outcome = _selection_outcome(
        checkpoint=checkpoint,
        best_reward=12.5,
        best_selection_score=18.0,
        best_rollout_iteration=3,
    )

    assert outcome["promotion_status"] == "CANDIDATE_PENDING_CPU_MUJOCO_EXAM"
    assert outcome["candidate_available"] is True
    assert outcome["candidate_checkpoint"] == checkpoint.name
    assert outcome["candidate_checkpoint_bytes"] == checkpoint.stat().st_size
    assert outcome["best_rollout_iteration"] == 3
    assert outcome["rejection_reasons"] == []


def test_candidate_without_rollout_binding_is_rejected(tmp_path) -> None:
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")

    with pytest.raises(ValueError, match="rollout iteration"):
        _selection_outcome(
            checkpoint=checkpoint,
            best_reward=1.0,
            best_selection_score=1.0,
            best_rollout_iteration=None,
        )


def test_truth_first_selection_never_forgets_a_real_save_for_proxy_reward() -> None:
    config = GoalkeeperPhysicsPPOConfig()
    saved = _candidate(first_save_rate=0.125, anatomical_selection_score=1.0)
    proxy_only = _candidate(
        first_save_rate=0.0,
        first_hand_save_rate=0.0,
        anatomical_selection_score=1_000.0,
    )

    assert _candidate_is_selectable(saved, config=config, best_truth_key=None)
    assert not _candidate_is_selectable(
        proxy_only,
        config=config,
        best_truth_key=_candidate_truth_key(saved),
    )


def test_exploration_resume_prefers_qualified_save_over_raw_contact() -> None:
    stable = _candidate(
        qualified_first_save_rate=0.20,
        first_save_rate=0.25,
        failed_rate=0.05,
    )
    flashy_but_falling = _candidate(
        qualified_first_save_rate=0.10,
        first_save_rate=0.80,
        failed_rate=0.70,
    )

    assert _exploration_truth_key(stable) > _exploration_truth_key(flashy_but_falling)


def test_save_first_exploration_retains_real_saves_even_when_keeper_falls() -> None:
    conservative = _candidate(
        first_save_rate=0.25,
        minimum_seed_first_save_rate=0.20,
        minimum_first_save_stratum_rate=0.12,
        minimum_first_save_side_rate=0.10,
        failed_rate=0.05,
    )
    diving = _candidate(
        first_save_rate=0.70,
        minimum_seed_first_save_rate=0.62,
        minimum_first_save_stratum_rate=0.48,
        minimum_first_save_side_rate=0.50,
        failed_rate=0.65,
    )

    assert _exploration_truth_key(diving, save_first=True) > _exploration_truth_key(
        conservative, save_first=True
    )
    assert _exploration_truth_key(conservative) > _exploration_truth_key(diving)


def test_exploration_and_safe_resume_rank_the_weakest_side_first() -> None:
    balanced = _candidate(
        qualified_first_save_rate=0.12,
        minimum_qualified_first_save_side_rate=0.10,
    )
    lopsided = _candidate(
        qualified_first_save_rate=0.30,
        minimum_qualified_first_save_side_rate=0.02,
    )

    assert _exploration_truth_key(balanced) > _exploration_truth_key(lopsided)
    assert _safe_continuation_truth_key(
        balanced, maximum_root_angular_speed_rad_s=3.5
    ) > _safe_continuation_truth_key(lopsided, maximum_root_angular_speed_rad_s=3.5)


def test_exploration_selection_uses_worst_seed_not_panel_mean() -> None:
    robust = _candidate(
        qualified_first_save_rate=0.10,
        minimum_qualified_first_save_side_rate=0.08,
        minimum_seed_qualified_first_save_rate=0.08,
        minimum_seed_qualified_first_save_side_rate=0.06,
        maximum_seed_failed_rate=0.10,
        minimum_seed_first_save_rate=0.20,
    )
    brittle = _candidate(
        qualified_first_save_rate=0.30,
        minimum_qualified_first_save_side_rate=0.20,
        minimum_seed_qualified_first_save_rate=0.02,
        minimum_seed_qualified_first_save_side_rate=0.01,
        maximum_seed_failed_rate=0.60,
        minimum_seed_first_save_rate=0.05,
    )

    assert _exploration_truth_key(robust) > _exploration_truth_key(brittle)
    assert not _candidate_is_selectable(
        _candidate(maximum_seed_failed_rate=0.01),
        config=GoalkeeperPhysicsPPOConfig(),
        best_truth_key=None,
    )


def test_truth_first_selection_uses_anatomical_score_only_as_tie_break() -> None:
    config = GoalkeeperPhysicsPPOConfig()
    incumbent = _candidate(anatomical_selection_score=12.0)
    better_tie = _candidate(anatomical_selection_score=12.1)

    assert _candidate_is_selectable(
        better_tie,
        config=config,
        best_truth_key=_candidate_truth_key(incumbent),
    )


def test_first_save_floor_rejects_non_effective_candidates() -> None:
    config = GoalkeeperPhysicsPPOConfig(minimum_first_save_rate_for_selection=0.80)

    assert not _candidate_is_selectable(
        _candidate(first_save_rate=0.79),
        config=config,
        best_truth_key=None,
    )
    assert _candidate_is_selectable(
        _candidate(first_save_rate=0.80),
        config=config,
        best_truth_key=None,
    )


def test_first_save_selection_floor_is_bounded() -> None:
    with pytest.raises(ValueError, match="first-save selection floor"):
        GoalkeeperPhysicsPPOConfig(minimum_first_save_rate_for_selection=1.01)

    with pytest.raises(ValueError, match="first-save stratum selection floor"):
        GoalkeeperPhysicsPPOConfig(minimum_first_save_stratum_rate_for_selection=1.01)
    with pytest.raises(ValueError, match="training unsafe penalty"):
        GoalkeeperPhysicsPPOConfig(training_unsafe_penalty=9.0)


def test_each_height_stratum_must_clear_its_own_selection_floor() -> None:
    config = GoalkeeperPhysicsPPOConfig(
        minimum_first_save_rate_for_selection=0.80,
        minimum_first_save_stratum_rate_for_selection=0.80,
    )

    assert not _candidate_is_selectable(
        _candidate(first_save_rate=0.90, minimum_first_save_stratum_rate=0.79),
        config=config,
        best_truth_key=None,
    )
    assert _candidate_is_selectable(
        _candidate(first_save_rate=0.85, minimum_first_save_stratum_rate=0.80),
        config=config,
        best_truth_key=None,
    )


@pytest.mark.parametrize("field", ["quarantined_rate", "nonfinite_quarantine_rate"])
def test_candidate_selection_rejects_any_quarantined_world(field: str) -> None:
    assert not _candidate_is_selectable(
        _candidate(**{field: 1.0 / 128.0}),
        config=GoalkeeperPhysicsPPOConfig(),
        best_truth_key=None,
    )


def test_failure_events_are_counted_per_hard_height_stratum() -> None:
    torch = pytest.importorskip("torch")
    environment = SimpleNamespace(
        _target_one=torch.tensor(
            [
                [0.0, 0.0, 0.40],
                [0.0, 0.0, 0.59],
                [0.0, 0.0, 0.60],
                [0.0, 0.0, 1.09],
                [0.0, 0.0, 1.10],
                [0.0, 0.0, 1.35],
            ]
        )
    )
    failed = torch.tensor([True, False, True, False, True, True])

    counts = _height_stratum_event_counts(torch, environment, failed)

    assert tuple(float(value) for value in counts) == (1.0, 1.0, 2.0)
