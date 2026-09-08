from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.growth.dynamic_strike_coordination import (
    STRIKE_COORDINATION_FEATURES,
    DynamicStrikeCoordinationActor,
    StrikeCoordinationObservation,
)
from rosclaw_soccer.training.dynamic_strike_coordination_exam import _actual_goal_crossing
from rosclaw_soccer.training.dynamic_strike_coordination_learning import (
    DynamicStrikeCoordinationDataset,
    fit_dynamic_strike_coordination_actor,
)
from rosclaw_soccer.training.dynamic_strike_coordination_probe import _project_shot
from rosclaw_soccer.world.field import G1TrainingGoalSpec


def _observation() -> StrikeCoordinationObservation:
    return StrikeCoordinationObservation(
        phase_progress=0.5,
        stance_depth_m=0.42,
        stance_lateral_error_m=0.31,
        stance_yaw_error_rad=0.72,
        approach_yaw_error_rad=0.24,
        ball_speed_mps=0.45,
    )


def test_constant_coordination_probe_has_only_bounded_blend_authority() -> None:
    actor = DynamicStrikeCoordinationActor.constant(
        stance_blend=0.30,
        goal_yaw_blend=0.40,
    )

    action = actor.act(_observation())

    assert action.stance_blend == pytest.approx(0.30)
    assert action.goal_yaw_blend == pytest.approx(0.40)
    assert actor.policy_type == "parameter"
    assert actor.dataset_snapshot_hash is None
    assert actor.actor_hash.startswith("sha256:")


def test_zero_constant_probe_stays_inside_actor_parameter_envelope() -> None:
    actor = DynamicStrikeCoordinationActor.constant(
        stance_blend=0.0,
        goal_yaw_blend=0.0,
    )

    action = actor.act(_observation())

    assert action.stance_blend < 0.00001
    assert action.goal_yaw_blend < 0.00001


def test_phase_ramp_changes_only_with_phase_progress() -> None:
    actor = DynamicStrikeCoordinationActor.phase_ramp(
        stance_start_blend=0.05,
        stance_end_blend=0.35,
        goal_yaw_start_blend=0.10,
        goal_yaw_end_blend=0.60,
    )
    early = actor.act(replace(_observation(), phase_progress=0.0))
    late = actor.act(replace(_observation(), phase_progress=1.0))

    assert early.stance_blend == pytest.approx(0.05)
    assert early.goal_yaw_blend == pytest.approx(0.10)
    assert late.stance_blend == pytest.approx(0.35)
    assert late.goal_yaw_blend == pytest.approx(0.60)
    assert sum(abs(value) > 0.0 for value in actor.weights) == 2


def test_learned_coordination_actor_requires_dataset_commitment() -> None:
    with pytest.raises(ValueError, match="SIM-only contract"):
        DynamicStrikeCoordinationActor(
            weights=(0.0,) * (2 * len(STRIKE_COORDINATION_FEATURES)),
            biases=(0.0, 0.0),
            policy_type="learned_linear",
        )


def test_coordination_actor_rejects_hardware_or_nonfinite_observation() -> None:
    with pytest.raises(ValueError, match="SIM-only contract"):
        DynamicStrikeCoordinationActor(
            weights=(0.0,) * (2 * len(STRIKE_COORDINATION_FEATURES)),
            biases=(0.0, 0.0),
            hardware_authorized=True,
        )
    with pytest.raises(ValueError, match="observation"):
        StrikeCoordinationObservation(
            phase_progress=0.5,
            stance_depth_m=math.nan,
            stance_lateral_error_m=0.2,
            stance_yaw_error_rad=0.2,
            approach_yaw_error_rad=0.2,
            ball_speed_mps=0.4,
        )


def test_probe_projects_post_contact_ballistics_without_keeper_contact() -> None:
    time = np.arange(20, dtype=np.float64) * 0.02
    pose = np.zeros((20, 7), dtype=np.float64)
    pose[:, 3] = 1.0
    pose[:, :3] = (5.0, 0.25, 0.50)
    velocity = np.zeros((20, 6), dtype=np.float64)
    velocity[:, :3] = (5.0, 0.50, 2.0)

    y_value, z_value, error, sample_frame = _project_shot(
        trajectory={"time": time, "ball_pose": pose, "ball_velocity": velocity},
        strike_frame=2,
        goal_plane_x_m=7.5,
        target_y_m=0.50,
        target_z_m=0.27375,
    )

    assert y_value == pytest.approx(0.50)
    assert z_value == pytest.approx(0.27375)
    assert error == pytest.approx(0.0)
    assert sample_frame == 2


def test_probe_projection_is_absent_without_physical_strike() -> None:
    assert _project_shot(
        trajectory={
            "time": np.asarray((0.0, 0.02)),
            "ball_pose": np.zeros((2, 7)),
            "ball_velocity": np.zeros((2, 6)),
        },
        strike_frame=-1,
        goal_plane_x_m=7.5,
        target_y_m=0.0,
        target_z_m=1.0,
    ) == (None, None, None, None)


def test_fit_produces_dataset_bound_learned_actor() -> None:
    observations = np.asarray(
        [
            (0.0, 0.40, 0.55, 0.90, 0.20, 0.45),
            (0.5, 0.38, 0.42, 0.65, 0.16, 0.40),
            (1.0, 0.36, 0.30, 0.35, 0.12, 0.30),
            (0.2, 0.42, 0.50, 0.80, 0.18, 0.42),
        ],
        dtype=np.float64,
    )
    dataset = DynamicStrikeCoordinationDataset(
        observations=observations,
        actions=np.asarray(((0.15, 0.22), (0.16, 0.24), (0.18, 0.28), (0.15, 0.23))),
        source_index=np.asarray((0, 0, 1, 1), dtype=np.int64),
        source_probe_paths=("/external/a", "/external/b"),
        source_report_hashes=("sha256:" + "1" * 64, "sha256:" + "2" * 64),
        source_trajectory_digests=("sha256:" + "3" * 64, "sha256:" + "4" * 64),
        dataset_snapshot_hash="sha256:" + "5" * 64,
    )

    actor, metrics = fit_dynamic_strike_coordination_actor(dataset)

    assert actor.policy_type == "learned_linear"
    assert actor.dataset_snapshot_hash == dataset.dataset_snapshot_hash
    assert metrics["maximum_absolute_error"] < 0.02
    assert actor.act(StrikeCoordinationObservation(*observations[0])).stance_blend > 0.0


def test_actual_goal_crossing_is_distinct_from_ballistic_intent() -> None:
    pose = np.zeros((3, 7), dtype=np.float64)
    pose[:, 3] = 1.0
    pose[:, :3] = ((7.0, 0.0, 0.4), (7.4, 0.2, 0.5), (7.8, 0.4, 0.6))

    crossing = _actual_goal_crossing(
        {"ball_pose": pose},
        goal=G1TrainingGoalSpec(plane_x_m=7.5, target_y_m=0.8),
    )

    assert crossing["frame"] == 1
    assert crossing["goal_plane_y_m"] == pytest.approx(0.25)
    assert crossing["goal_plane_z_m"] == pytest.approx(0.525)
    assert crossing["inside"] is True
