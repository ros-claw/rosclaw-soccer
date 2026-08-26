from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.training.goalkeeper_multistep import (
    GoalkeeperEpisodePhase,
    GoalkeeperMultiStepAccumulator,
    GoalkeeperMultiStepConfig,
    GoalkeeperStepBatch,
)


def _step(
    *,
    time: float,
    shot: int,
    contact: bool = False,
    hand_contact: bool = False,
    save: bool = False,
    pelvis: float = 0.80,
    upright: float = 1.0,
    speed: float = 0.0,
    angular: float = 0.0,
    action: float = 0.0,
    previous: float = 0.0,
    posture_exception: bool = False,
) -> GoalkeeperStepBatch:
    return GoalkeeperStepBatch(
        time_sec=np.asarray([time], dtype=np.float64),
        ball_position_m=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        ball_velocity_mps=np.asarray([[4.0, 0.0, 0.0]], dtype=np.float64),
        intercept_target_m=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        left_hand_position_m=np.asarray([[0.0, 0.05, 1.0]], dtype=np.float64),
        right_hand_position_m=np.asarray([[0.0, -0.05, 1.0]], dtype=np.float64),
        pelvis_height_m=np.asarray([pelvis], dtype=np.float64),
        root_linear_velocity_mps=np.asarray([[speed, 0.0, 0.0]], dtype=np.float64),
        root_angular_velocity_rad_s=np.asarray([[0.0, angular, 0.0]], dtype=np.float64),
        upright_projection=np.asarray([upright], dtype=np.float64),
        action=np.full((1, 29), action, dtype=np.float64),
        previous_action=np.full((1, 29), previous, dtype=np.float64),
        joint_acceleration_rad_s2=np.zeros((1, 29), dtype=np.float64),
        applied_torque_nm=np.zeros((1, 29), dtype=np.float64),
        ball_contact=np.asarray([contact], dtype=np.bool_),
        hand_contact=np.asarray([hand_contact], dtype=np.bool_),
        true_save=np.asarray([save], dtype=np.bool_),
        shot_index=np.asarray([shot], dtype=np.int64),
        posture_exception_granted=np.asarray([posture_exception], dtype=np.bool_),
    )


def test_numpy_posture_exception_matches_bounded_torch_semantics() -> None:
    task = GoalkeeperMultiStepAccumulator(1)
    dynamic = task.step(
        _step(
            time=0.5,
            shot=1,
            pelvis=0.50,
            upright=0.50,
            posture_exception=True,
        )
    )
    assert dynamic.phase[0] == GoalkeeperEpisodePhase.FIRST_FLIGHT
    assert not bool(dynamic.terminated[0])

    expired = task.step(_step(time=0.52, shot=1, pelvis=0.50, upright=0.50))
    assert expired.phase[0] == GoalkeeperEpisodePhase.FAILED
    assert bool(expired.terminated[0])


def test_multistep_requires_recovery_before_second_save() -> None:
    config = GoalkeeperMultiStepConfig(recovery_hold_sec=0.04)
    task = GoalkeeperMultiStepAccumulator(1, config)

    first = task.step(_step(time=0.5, shot=1, contact=True, save=True))
    assert bool(first.first_save[0])
    assert first.phase[0] == GoalkeeperEpisodePhase.FIRST_IMPACT

    task.step(_step(time=0.52, shot=0))
    recovered = task.step(_step(time=0.54, shot=0))
    assert bool(recovered.recovered_after_first[0])
    assert recovered.event_bonus[0] >= config.recovery_bonus

    second = task.step(_step(time=2.8, shot=2, contact=True, save=True))
    assert bool(second.second_save[0])
    assert bool(second.second_attempt_save[0])
    assert second.event_bonus[0] >= config.contact_bonus + config.second_save_bonus
    assert task.summary()["second_save_rate"] == 1.0
    assert task.summary()["promotion_status"] == "TRAINING_METRICS_ONLY_NOT_PROMOTED"


def test_second_collision_while_falling_does_not_satisfy_second_save() -> None:
    task = GoalkeeperMultiStepAccumulator(1)
    task.step(_step(time=0.5, shot=1, contact=True, save=True))
    task.step(_step(time=2.8, shot=2, contact=False, save=False, speed=1.0))
    second = task.step(_step(time=2.82, shot=2, contact=True, save=True, speed=1.0))

    assert not bool(second.recovered_after_first[0])
    assert bool(second.second_attempt_save[0])
    assert not bool(second.second_save[0])
    assert task.summary()["second_attempt_save_rate"] == 1.0
    assert second.event_bonus[0] < task.config.second_save_bonus


def test_contact_without_a_save_does_not_start_recovery_credit() -> None:
    config = GoalkeeperMultiStepConfig(recovery_hold_sec=0.04)
    task = GoalkeeperMultiStepAccumulator(1, config)
    task.step(_step(time=0.5, shot=1, contact=True, save=False))
    task.step(_step(time=0.52, shot=0))
    result = task.step(_step(time=0.54, shot=0))

    assert not bool(result.recovered_after_first[0])
    assert result.event_bonus[0] < config.recovery_bonus


def test_multistep_smooth_actions_and_real_saves_outscore_twitches() -> None:
    smooth_task = GoalkeeperMultiStepAccumulator(1)
    twitch_task = GoalkeeperMultiStepAccumulator(1)

    smooth = smooth_task.step(_step(time=0.5, shot=1, contact=True, save=True))
    twitch = twitch_task.step(
        _step(time=0.5, shot=1, contact=True, save=False, action=1.0, previous=-1.0)
    )

    assert smooth.total[0] > twitch.total[0]
    assert twitch.smoothness_penalty[0] > smooth.smoothness_penalty[0]


def test_multistep_rewards_anatomical_hand_save_over_body_block() -> None:
    body_task = GoalkeeperMultiStepAccumulator(1)
    hand_task = GoalkeeperMultiStepAccumulator(1)

    body = body_task.step(_step(time=0.5, shot=1, contact=True, save=True))
    hand = hand_task.step(_step(time=0.5, shot=1, contact=True, hand_contact=True, save=True))

    assert hand.total[0] > body.total[0]
    assert bool(hand.first_hand_save[0])
    assert hand_task.summary()["first_hand_save_rate"] == 1.0


def test_multistep_densifies_second_shot_reach_without_relaxing_double_save() -> None:
    config = GoalkeeperMultiStepConfig(second_shot_reach_reward_multiplier=1.6)
    first = GoalkeeperMultiStepAccumulator(1, config).step(_step(time=0.5, shot=1))
    second = GoalkeeperMultiStepAccumulator(1, config).step(_step(time=2.8, shot=2))

    assert second.reach[0] == pytest.approx(1.6 * first.reach[0])
    assert not bool(second.second_save[0])


def test_multistep_adds_long_range_signal_only_for_hard_height() -> None:
    config = GoalkeeperMultiStepConfig(hard_height_reach_reward_scale=2.0)
    low_sample = _step(time=0.5, shot=1)
    high_sample = replace(
        low_sample,
        intercept_target_m=np.asarray([[0.0, 0.0, 1.30]], dtype=np.float64),
    )
    closer_high = replace(
        high_sample,
        left_hand_position_m=np.asarray([[0.0, 0.05, 1.20]], dtype=np.float64),
        right_hand_position_m=np.asarray([[0.0, -0.05, 1.20]], dtype=np.float64),
    )
    shaped_task = GoalkeeperMultiStepAccumulator(1, config)
    baseline_task = GoalkeeperMultiStepAccumulator(1)
    shaped_task.step(high_sample)
    baseline_task.step(high_sample)
    high = shaped_task.step(closer_high)
    base_high = baseline_task.step(closer_high)
    low = GoalkeeperMultiStepAccumulator(1, config).step(low_sample)

    assert high.reach[0] > base_high.reach[0]
    assert low.reach[0] == pytest.approx(
        GoalkeeperMultiStepAccumulator(1).step(low_sample).reach[0]
    )


def test_multistep_hard_height_reward_cannot_be_farmed_by_holding() -> None:
    config = GoalkeeperMultiStepConfig(hard_height_reach_reward_scale=10.0)
    sample = replace(
        _step(time=0.5, shot=1),
        intercept_target_m=np.asarray([[0.0, 0.0, 1.30]], dtype=np.float64),
    )
    shaped_task = GoalkeeperMultiStepAccumulator(1, config)
    baseline_task = GoalkeeperMultiStepAccumulator(1)
    shaped_task.step(sample)
    baseline_task.step(sample)
    held = shaped_task.step(sample)
    baseline_held = baseline_task.step(sample)

    assert held.reach[0] == pytest.approx(baseline_held.reach[0])


def test_multistep_potential_reach_cannot_be_farmed_by_holding() -> None:
    config = GoalkeeperMultiStepConfig(reach_reward_semantics="POTENTIAL_PROGRESS_ONLY")
    task = GoalkeeperMultiStepAccumulator(1, config)
    far = replace(
        _step(time=0.5, shot=1),
        left_hand_position_m=np.asarray([[0.0, 0.80, 1.0]], dtype=np.float64),
        right_hand_position_m=np.asarray([[0.0, -0.80, 1.0]], dtype=np.float64),
    )
    assert task.step(far).reach[0] == pytest.approx(0.0)
    assert task.step(far).reach[0] == pytest.approx(0.0)

    closer = replace(
        far,
        left_hand_position_m=np.asarray([[0.0, 0.20, 1.0]], dtype=np.float64),
        right_hand_position_m=np.asarray([[0.0, -0.20, 1.0]], dtype=np.float64),
    )
    assert task.step(closer).reach[0] > 0.0
    assert task.step(far).reach[0] < 0.0
    with pytest.raises(ValueError, match="reward semantics"):
        replace(config, reach_reward_semantics="ABSOLUTE_DISTANCE")


def test_multistep_task_motion_rewards_safe_body_progress_not_holding() -> None:
    config = GoalkeeperMultiStepConfig(task_motion_reward_scale=8.0)
    task = GoalkeeperMultiStepAccumulator(1, config)
    far = replace(
        _step(time=0.5, shot=1),
        intercept_target_m=np.asarray([[4.49, 0.90, 0.20]], dtype=np.float64),
        pelvis_position_m=np.asarray([[4.52, 0.00, 0.793]], dtype=np.float64),
    )
    assert task.step(far).task_motion[0] == pytest.approx(0.0)

    closer = replace(
        far,
        pelvis_height_m=np.asarray([0.60], dtype=np.float64),
        pelvis_position_m=np.asarray([[4.52, 0.45, 0.60]], dtype=np.float64),
        posture_exception_granted=np.asarray([True], dtype=np.bool_),
    )
    assert task.step(closer).task_motion[0] > 0.0
    assert task.step(closer).task_motion[0] == pytest.approx(0.0)
    assert task.step(far).task_motion[0] < 0.0


def test_multistep_task_motion_fails_closed_without_pelvis_position() -> None:
    task = GoalkeeperMultiStepAccumulator(
        1, GoalkeeperMultiStepConfig(task_motion_reward_scale=1.0)
    )
    with pytest.raises(ValueError, match="requires pelvis position"):
        task.step(_step(time=0.5, shot=1))


def test_multistep_task_motion_arrival_bonus_couples_body_hand_and_deadline() -> None:
    config = GoalkeeperMultiStepConfig(task_motion_reward_scale=8.0)
    prepared = replace(
        _step(time=0.5, shot=1),
        ball_position_m=np.asarray([[3.00, 0.90, 0.20]], dtype=np.float64),
        intercept_target_m=np.asarray([[4.49, 0.90, 0.20]], dtype=np.float64),
        pelvis_height_m=np.asarray([0.60], dtype=np.float64),
        pelvis_position_m=np.asarray([[4.52, 0.45, 0.60]], dtype=np.float64),
        posture_exception_granted=np.asarray([True], dtype=np.bool_),
    )
    aligned = replace(
        prepared,
        ball_position_m=np.asarray([[4.05, 0.90, 0.20]], dtype=np.float64),
        left_hand_position_m=np.asarray([[4.49, 0.88, 0.20]], dtype=np.float64),
        right_hand_position_m=np.asarray([[4.49, 0.70, 0.20]], dtype=np.float64),
    )
    late = replace(
        aligned,
        ball_position_m=np.asarray([[4.60, 0.90, 0.20]], dtype=np.float64),
    )

    aligned_task = GoalkeeperMultiStepAccumulator(1, config)
    aligned_task.step(prepared)
    arrival_reward = aligned_task.step(aligned).task_motion[0]
    held_reward = aligned_task.step(aligned).task_motion[0]

    unaligned_task = GoalkeeperMultiStepAccumulator(1, config)
    unaligned_task.step(prepared)
    unaligned_reward = unaligned_task.step(
        replace(
            prepared,
            ball_position_m=aligned.ball_position_m,
        )
    ).task_motion[0]
    late_task = GoalkeeperMultiStepAccumulator(1, config)
    late_task.step(prepared)

    assert arrival_reward > unaligned_reward
    assert arrival_reward > 0.0
    assert unaligned_reward == pytest.approx(0.0)
    assert held_reward == pytest.approx(0.0)
    assert late_task.step(late).task_motion[0] == pytest.approx(0.0)


def test_multistep_rejects_hand_contact_without_robot_contact() -> None:
    with pytest.raises(ValueError, match="hand contact"):
        GoalkeeperMultiStepAccumulator(1).step(
            _step(time=0.5, shot=1, contact=False, hand_contact=True)
        )


def test_multistep_fast_body_rotation_is_penalized() -> None:
    task_one = GoalkeeperMultiStepAccumulator(1)
    task_two = GoalkeeperMultiStepAccumulator(1)
    stable = task_one.step(_step(time=0.5, shot=1, angular=0.0))
    spinning = task_two.step(_step(time=0.5, shot=1, angular=2.0))
    assert spinning.total[0] < stable.total[0]
    assert spinning.smoothness_penalty[0] > stable.smoothness_penalty[0]


def test_multistep_tail_penalty_targets_only_angular_excess() -> None:
    baseline = GoalkeeperMultiStepAccumulator(
        1,
        GoalkeeperMultiStepConfig(root_angular_speed_soft_limit_rad_s=2.0),
    ).step(_step(time=0.5, shot=1, angular=3.0))
    tail = GoalkeeperMultiStepAccumulator(
        1,
        GoalkeeperMultiStepConfig(
            root_angular_speed_soft_limit_rad_s=2.0,
            root_angular_speed_excess_penalty_scale=10.0,
        ),
    ).step(_step(time=0.5, shot=1, angular=3.0))

    assert tail.smoothness_penalty[0] - baseline.smoothness_penalty[0] == pytest.approx(10.0)
    with pytest.raises(ValueError, match="root-angular tail penalty"):
        GoalkeeperMultiStepConfig(root_angular_speed_excess_penalty_scale=101.0)


def test_multistep_recovery_potential_rewards_braking_after_save() -> None:
    config = GoalkeeperMultiStepConfig(
        recovery_progress_reward_scale=20.0,
        recovery_progress_linear_speed_decay=2.0,
        recovery_progress_angular_speed_decay=0.5,
    )
    braking = GoalkeeperMultiStepAccumulator(1, config)
    worsening = GoalkeeperMultiStepAccumulator(1, config)
    braking.step(_step(time=0.50, shot=1, contact=True, save=True, angular=2.0))
    worsening.step(_step(time=0.50, shot=1, contact=True, save=True, angular=2.0))

    improved = braking.step(_step(time=0.52, shot=0, angular=1.0))
    degraded = worsening.step(_step(time=0.52, shot=0, angular=3.0))

    assert improved.recovery_progress[0] > 0.0
    assert degraded.recovery_progress[0] < 0.0
    assert improved.total[0] > degraded.total[0]


def test_multistep_fails_closed_on_fall_and_validates_shapes() -> None:
    task = GoalkeeperMultiStepAccumulator(1)
    fallen = task.step(_step(time=0.3, shot=1, pelvis=0.30, upright=0.2))
    assert bool(fallen.terminated[0])
    assert fallen.phase[0] == GoalkeeperEpisodePhase.FAILED
    assert fallen.safety_penalty[0] > 0.0

    invalid = replace(_step(time=0.2, shot=1), action=np.zeros((1, 28)))
    with pytest.raises(ValueError, match="current and previous action widths"):
        task.step(invalid)


def test_multistep_charges_extra_debt_when_a_save_ends_unsafe() -> None:
    baseline = GoalkeeperMultiStepAccumulator(1)
    debt = GoalkeeperMultiStepAccumulator(
        1,
        GoalkeeperMultiStepConfig(save_then_unsafe_penalty=125.0),
    )
    for task in (baseline, debt):
        task.step(_step(time=0.50, shot=1, contact=True, save=True))

    baseline_failure = baseline.step(_step(time=0.52, shot=0, pelvis=0.30, upright=0.2))
    debt_failure = debt.step(_step(time=0.52, shot=0, pelvis=0.30, upright=0.2))

    assert debt_failure.safety_penalty[0] - baseline_failure.safety_penalty[0] == 125.0
    assert debt_failure.total[0] - baseline_failure.total[0] == -125.0
    with pytest.raises(ValueError, match="save-then-unsafe"):
        GoalkeeperMultiStepConfig(save_then_unsafe_penalty=-1.0)


def test_multistep_failed_phase_cannot_be_laundered_by_safe_timeout() -> None:
    task = GoalkeeperMultiStepAccumulator(1)
    failed = task.step(_step(time=0.3, shot=1, pelvis=0.30, upright=0.2))
    assert failed.phase[0] == GoalkeeperEpisodePhase.FAILED

    after_quarantine = task.step(
        _step(time=task.config.episode_duration_sec, shot=0, pelvis=0.80, upright=1.0)
    )

    assert after_quarantine.phase[0] == GoalkeeperEpisodePhase.FAILED
    assert bool(after_quarantine.terminated[0])
    assert task.summary()["failed_rate"] == 1.0


def test_multistep_failed_save_cannot_reenter_recovery() -> None:
    task = GoalkeeperMultiStepAccumulator(1)
    task.step(_step(time=0.5, shot=1, contact=True, hand_contact=True, save=True))
    failed = task.step(_step(time=0.6, shot=0, pelvis=0.30, upright=0.2))
    assert failed.phase[0] == GoalkeeperEpisodePhase.FAILED

    after_quarantine = task.step(_step(time=0.7, shot=0, pelvis=0.80, upright=1.0))

    assert after_quarantine.phase[0] == GoalkeeperEpisodePhase.FAILED
    assert not bool(after_quarantine.recovered_after_first[0])


def test_multistep_supports_hierarchical_action_and_joint_widths() -> None:
    sample = replace(
        _step(time=0.2, shot=1),
        action=np.zeros((1, 18)),
        previous_action=np.zeros((1, 18)),
    )
    result = GoalkeeperMultiStepAccumulator(1).step(sample)
    assert result.total.shape == (1,)


def test_multistep_configuration_is_content_bound_and_sim_only() -> None:
    config = GoalkeeperMultiStepConfig()
    assert config.config_hash.startswith("sha256:")
    assert config.recovery_hold_steps == 12
    with pytest.raises(ValueError, match="SIM_ONLY"):
        GoalkeeperMultiStepConfig(hardware_authorized=True)
