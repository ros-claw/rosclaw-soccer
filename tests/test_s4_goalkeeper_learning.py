from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.skills.team.goalkeeper_learning import (
    GoalkeeperBlockSearchConfig,
    _block_trial,
    goalkeeper_block_parent_config,
)
from rosclaw_soccer.skills.team.shared_world import (
    G1GoalkeeperConfig,
    G1SharedWorldResult,
    _goalkeeper_glove_geoms,
    _goalkeeper_neutral_root_pose,
)
from rosclaw_soccer.world.field import build_g1_three_player_stadium_model


def _result(**overrides: object) -> G1SharedWorldResult:
    values: dict[str, object] = {
        "finite_state": True,
        "pass_contact_observed": True,
        "shot_contact_observed": True,
        "pass_contact_time_sec": 1.0,
        "shot_contact_time_sec": 2.0,
        "pass_peak_ball_speed_mps": 1.0,
        "shot_peak_ball_speed_mps": 8.0,
        "goal_crossed": False,
        "goal_plane_crossed": False,
        "goal_crossing_y_m": None,
        "goal_crossing_z_m": None,
        "target_error_m": None,
        "passer_min_pelvis_height_m": 0.75,
        "shooter_min_pelvis_height_m": 0.75,
        "passer_roll_peak_rad": 0.1,
        "passer_pitch_peak_rad": 0.1,
        "shooter_roll_peak_rad": 0.1,
        "shooter_pitch_peak_rad": 0.1,
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
        "goalkeeper_min_pelvis_height_m": 0.73,
        "goalkeeper_ball_contact_observed": True,
        "goalkeeper_ball_contact_time_sec": 3.0,
        "goalkeeper_save_observed": True,
    }
    values.update(overrides)
    return G1SharedWorldResult(**values)  # type: ignore[arg-type]


def _trajectory() -> dict[str, np.ndarray]:
    time = np.arange(0.0, 4.0, 0.02)
    velocity = np.zeros((len(time), 6), dtype=np.float64)
    contact = int(np.searchsorted(time, 3.0))
    velocity[:contact, 0] = 8.0
    velocity[contact:, 0] = 5.0
    return {
        "time": time,
        "ball_velocity": velocity,
        "digest_padding": np.zeros((len(time), 1), dtype=np.float64),
    }


def test_goalkeeper_parent_is_early_causal_and_block_residual_is_off() -> None:
    config = goalkeeper_block_parent_config(G1GoalkeeperConfig())

    assert config.anticipation_enabled
    assert config.anticipation_start_policy_frame == 180
    assert config.anticipation_maximum_foot_ball_distance_m == 4.0
    assert config.anticipation_velocity_scale == 1.0
    assert not config.block_action_enabled


def test_block_trial_accepts_a_dissipative_safe_save() -> None:
    candidate = replace(
        goalkeeper_block_parent_config(G1GoalkeeperConfig()),
        block_action_enabled=True,
        block_action_hip_pitch_rad=0.265,
    )
    trial = _block_trial(
        candidate=candidate,
        result=_result(),
        trajectory=_trajectory(),
        search_config=GoalkeeperBlockSearchConfig(),
    )

    assert trial.eligible
    assert trial.goalkeeper_save_observed
    assert trial.post_contact_speed_ratio < 0.80


def test_block_trial_rejects_an_unsafe_or_scoring_contact() -> None:
    candidate = replace(
        goalkeeper_block_parent_config(G1GoalkeeperConfig()),
        block_action_enabled=True,
        block_action_hip_pitch_rad=0.255,
    )
    unsafe = _block_trial(
        candidate=candidate,
        result=_result(goal_crossed=True, goal_plane_crossed=True, goalkeeper_save_observed=False),
        trajectory=_trajectory(),
        search_config=GoalkeeperBlockSearchConfig(),
    )
    low = _block_trial(
        candidate=candidate,
        result=_result(goalkeeper_min_pelvis_height_m=0.10),
        trajectory=_trajectory(),
        search_config=GoalkeeperBlockSearchConfig(),
    )

    assert not unsafe.eligible
    assert not low.eligible
    assert low.safety_cost > 0.0


def test_block_search_rejects_invalid_candidates() -> None:
    with pytest.raises(ValueError, match="unique"):
        GoalkeeperBlockSearchConfig(hip_pitch_candidates_rad=(0.1, 0.1, 0.2))


def test_goalkeeper_spawn_can_use_an_independent_neutral_root_pose() -> None:
    """Lock the root-frame bug that rotated lateral commands into depth drift."""

    position, quaternion = _goalkeeper_neutral_root_pose(
        np.asarray((0.15, -0.20, 0.77), dtype=np.float64)
    )

    assert np.allclose(position, (0.0, 0.0, 0.77), atol=1e-12)
    assert np.allclose(quaternion, (1.0, 0.0, 0.0, 0.0), atol=1e-12)


def test_shared_world_binds_the_anatomical_goalkeeper_gloves() -> None:
    asset_value = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if asset_value is None:
        pytest.skip("qualified G1 assets are unavailable")
    model = build_g1_three_player_stadium_model(
        Path(asset_value),
        passer_origin_m=(5.10, -0.164, 0.0),
        goalkeeper_origin_m=(7.02, 0.0, 0.0),
    )
    goalkeeper = SimpleNamespace(
        left_hand_body=int(model.body("goalkeeper_left_wrist_yaw_link").id),
        right_hand_body=int(model.body("goalkeeper_right_wrist_yaw_link").id),
    )

    resolved = _goalkeeper_glove_geoms(model=model, goalkeeper=goalkeeper)

    expected = {
        int(model.geom("goalkeeper_left_goalkeeper_glove").id),
        int(model.geom("goalkeeper_right_goalkeeper_glove").id),
    }
    assert resolved == expected
