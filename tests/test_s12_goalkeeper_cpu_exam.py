from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.training.goalkeeper_cpu_exam import (
    GoalkeeperCPUExamConfig,
    _miss_diagnostics,
    _require_declared_difficulty_world,
    _sample_shot,
)
from rosclaw_soccer.training.goalkeeper_mjwarp import (
    GoalkeeperMJWarpConfig,
    goalkeeper_world_config,
)


def test_cpu_exam_contract_is_content_bound_and_sim_only() -> None:
    config = GoalkeeperCPUExamConfig(episode_count=16)
    assert config.config_hash.startswith("sha256:")
    assert config.minimum_second_save_improvement > 0.0
    assert config.maximum_save_rate_regression == 0.0
    assert config.minimum_pelvis_height_m >= 0.65
    assert config.maximum_root_angular_speed_rad_s <= 3.5
    assert config.maximum_p95_root_angular_speed_rad_s < config.maximum_root_angular_speed_rad_s
    assert config.minimum_lateral_speed_improvement_mps > 0.0
    assert config.minimum_hand_displacement_improvement_m > 0.0
    assert config.maximum_p95_hand_speed_mps <= 5.0
    assert config.maximum_mean_second_release_lateral_error_m <= 0.55
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)


def test_cpu_exam_contract_rejects_weak_or_invalid_exam() -> None:
    with pytest.raises(ValueError, match="episode count"):
        GoalkeeperCPUExamConfig(episode_count=4)
    with pytest.raises(ValueError, match="action-step"):
        GoalkeeperCPUExamConfig(maximum_applied_actor_action_step=0.001)
    with pytest.raises(ValueError, match="root-angular-speed"):
        GoalkeeperCPUExamConfig(maximum_root_angular_speed_rad_s=0.1)
    with pytest.raises(ValueError, match="non-inferiority"):
        GoalkeeperCPUExamConfig(maximum_save_rate_regression=0.10)
    with pytest.raises(ValueError, match="first-save floor"):
        GoalkeeperCPUExamConfig(minimum_first_save_rate=1.01)
    target = GoalkeeperCPUExamConfig(
        minimum_first_save_rate=0.80,
        minimum_second_attempt_save_rate=0.80,
    )
    assert target.minimum_first_save_rate == pytest.approx(0.80)
    assert target.minimum_second_attempt_save_rate == pytest.approx(0.80)
    from rosclaw_soccer.training.goalkeeper_mobility_option import (
        GoalkeeperMobilityOptionConfig,
    )

    with pytest.raises(ValueError, match="residual plasticity"):
        GoalkeeperMobilityOptionConfig(residual_plasticity_scale=1.01)


def test_difficulty_factory_changes_the_physical_shot_envelope() -> None:
    mislabeled = GoalkeeperMJWarpConfig(environment_count=1, difficulty_profile="elite")
    elite = goalkeeper_world_config(difficulty_profile="elite", environment_count=1)

    assert mislabeled.config_hash != elite.config_hash
    assert elite.flight_time_range_sec == (0.30, 0.50)
    assert elite.target_z_range_m == (0.10, 1.65)
    with pytest.raises(ValueError, match="does not match its declared difficulty"):
        _require_declared_difficulty_world(mislabeled)
    _require_declared_difficulty_world(elite)


def test_hard_shot_curriculum_targets_high_far_corners() -> None:
    world = goalkeeper_world_config(
        difficulty_profile="match",
        environment_count=1,
        shot_intent_cue_enabled=True,
        hard_shot_fraction=0.9,
    )
    rng = np.random.default_rng(123)
    shots = [_sample_shot(rng, world) for _ in range(512)]
    hard = [
        shot
        for shot in shots
        if abs(float(shot["target"][1])) >= 0.72 and float(shot["target"][2]) >= 1.10
    ]
    assert len(hard) / len(shots) > 0.85


def test_all_hard_balanced_curriculum_trains_far_low_mid_and_high() -> None:
    world = goalkeeper_world_config(
        difficulty_profile="match",
        environment_count=1,
        shot_intent_cue_enabled=True,
        hard_shot_fraction=1.0,
        hard_shot_height_mode="balanced",
    )
    rng = np.random.default_rng(321)
    shots = [_sample_shot(rng, world) for _ in range(900)]
    targets = np.asarray([shot["target"] for shot in shots])
    assert np.all(np.abs(targets[:, 1]) >= 0.72)
    bands = (
        int(np.count_nonzero(targets[:, 2] <= 0.60)),
        int(np.count_nonzero((targets[:, 2] > 0.60) & (targets[:, 2] <= 1.10))),
        int(np.count_nonzero(targets[:, 2] > 1.10)),
    )
    assert all(250 <= count <= 350 for count in bands)


def test_all_hard_low_specialist_curriculum_contains_only_far_low_shots() -> None:
    world = goalkeeper_world_config(
        difficulty_profile="match",
        environment_count=1,
        shot_intent_cue_enabled=True,
        hard_shot_fraction=1.0,
        hard_shot_height_mode="low",
    )
    rng = np.random.default_rng(322)
    shots = [_sample_shot(rng, world) for _ in range(256)]
    targets = np.asarray([shot["target"] for shot in shots])
    assert np.all(np.abs(targets[:, 1]) >= 0.72)
    assert np.all(targets[:, 2] <= 0.60)


def test_all_hard_scaffold_changes_time_not_far_corner_content() -> None:
    world = goalkeeper_world_config(
        difficulty_profile="match",
        environment_count=1,
        shot_intent_cue_enabled=True,
        hard_shot_fraction=1.0,
        hard_shot_height_mode="balanced",
        hard_shot_flight_time_range_sec=(0.70, 0.90),
    )
    rng = np.random.default_rng(4101)
    shots = [_sample_shot(rng, world) for _ in range(300)]
    assert all(abs(float(shot["target"][1])) >= 0.72 for shot in shots)
    assert all(0.70 <= float(shot["flight"]) <= 0.90 for shot in shots)
    _require_declared_difficulty_world(world)


def test_miss_diagnostics_separates_deployment_command_sign_from_world_motion() -> None:
    trajectory = {
        "time": np.asarray((1.0, 1.2)),
        "qpos": np.asarray(((0.0, 0.10), (0.0, 0.30))),
        "action": np.asarray(((-0.8,), (-0.6,))),
        "shot_index": np.asarray((1, 1)),
        "left_hand": np.asarray(((4.44, 0.25, 0.80), (4.44, 0.55, 0.80))),
        "right_hand": np.asarray(((4.44, -0.25, 0.80), (4.44, -0.15, 0.80))),
        "first_arrival_sec": np.asarray((1.2,)),
        "second_arrival_sec": np.asarray((3.5,)),
    }
    diagnostics = _miss_diagnostics(
        trajectory,
        target=np.asarray((4.44, 0.75, 0.80)),
        phase=1,
    )
    assert diagnostics["hand_distance_at_arrival_m"] == pytest.approx(0.20)
    assert diagnostics["root_lateral_error_at_arrival_m"] == pytest.approx(0.45)
    assert diagnostics["command_sign_contract_alignment"] == 1.0
    assert diagnostics["root_motion_target_alignment"] == 1.0
