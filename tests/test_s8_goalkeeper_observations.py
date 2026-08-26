from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.skills.goalkeeper_v2.observations import (
    GoalkeeperActorObserver,
    GoalkeeperObservationSpec,
)


def _observe(observer: GoalkeeperActorObserver, step: int, ball_x: float):
    zeros3 = np.zeros(3, dtype=np.float64)
    zeros29 = np.zeros(29, dtype=np.float64)
    return observer.observe(
        timestamp_sec=step * 0.02,
        ball_relative_position_m=np.asarray((ball_x, 0.0, 0.3), dtype=np.float64),
        gravity_orientation=np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
        root_linear_velocity_mps=zeros3,
        angular_velocity_rad_s=zeros3,
        joint_position_rad=zeros29,
        joint_velocity_rad_s=zeros29,
        previous_action_rad=zeros29,
    )


def test_actor_contract_has_no_shooter_phase_or_privileged_truth() -> None:
    spec = GoalkeeperObservationSpec()
    names = spec.actor_names

    assert not any("policy_frame" in name or "shooter" in name for name in names)
    assert not any("future" in name or "ground_truth" in name for name in names)
    assert "causal_ball_velocity_estimate.x" in names
    assert "causal_intercept_estimate.time" in names
    assert "ball_velocity.x" in spec.privileged_critic_names
    assert "intercept_time" in spec.privileged_critic_names
    assert spec.actor_contract_hash != spec.critic_contract_hash


def test_observer_detects_flight_from_visible_ball_history_only() -> None:
    observer = GoalkeeperActorObserver(flight_velocity_threshold_mps=0.10)
    stationary = _observe(observer, 0, 3.0)
    moving = _observe(observer, 1, 2.98)

    assert stationary.observed_flight_start_sec is None
    assert moving.estimated_ball_velocity_mps[0] < -0.10
    assert moving.observed_flight_start_sec == 0.02
    assert not moving.ball_history_ready
    assert moving.intercept_confidence > 0.0
    assert moving.estimated_intercept[0] > 0.0
    assert sum(moving.estimated_target_region) == pytest.approx(1.0)

    last = moving
    # The detection frame segments any pre-threat samples; eight subsequent
    # samples are therefore required before the new threat history is full.
    for step in range(2, 9):
        last = _observe(observer, step, 2.98 - 0.02 * step)
    assert last.ball_history_ready
    assert len(last.values) == len(observer.spec.actor_names)


def test_observer_does_not_mistake_a_ball_moving_away_for_a_shot() -> None:
    observer = GoalkeeperActorObserver(flight_velocity_threshold_mps=0.10)
    _observe(observer, 0, 1.0)
    moving_away = _observe(observer, 1, 1.1)

    assert moving_away.estimated_ball_velocity_mps[0] > 0.10
    assert moving_away.observed_flight_start_sec is None


def test_observer_compensates_keeper_ego_motion() -> None:
    observer = GoalkeeperActorObserver(flight_velocity_threshold_mps=0.10)
    zeros3 = np.zeros(3, dtype=np.float64)
    zeros29 = np.zeros(29, dtype=np.float64)
    _observe(observer, 0, 3.0)
    observation = observer.observe(
        timestamp_sec=0.02,
        ball_relative_position_m=np.asarray((3.0, 0.0, 0.3), dtype=np.float64),
        gravity_orientation=np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
        root_linear_velocity_mps=np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        angular_velocity_rad_s=zeros3,
        joint_position_rad=zeros29,
        joint_velocity_rad_s=zeros29,
        previous_action_rad=zeros29,
    )

    assert observation.estimated_ball_velocity_mps[0] == pytest.approx(0.0)
    assert observation.observed_flight_start_sec is None
    assert observation.intercept_confidence == pytest.approx(0.0)
    assert observation.estimated_target_region[-1] == pytest.approx(1.0)
