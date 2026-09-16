"""Explicit half-turn navigation frames preserve identity and physical boundaries."""

import math
from dataclasses import asdict, replace

import numpy as np
import pytest

from rosclaw_soccer.skills.team.navigation_option import NavigationDelta, NavigationObservation
from rosclaw_soccer.training.local_navigation import local_navigation_features
from rosclaw_soccer.training.navigation_frame import (
    canonical_navigation_delta,
    canonical_navigation_observation,
)


def observation() -> NavigationObservation:
    return NavigationObservation(
        agent_id="red.finisher",
        frame=20,
        time_sec=0.4,
        role="finisher",
        intent="shoot",
        body_pose=(1.0, 0.5, 0.75, 1.0, 0.0, 0.0, 0.0),
        body_velocity=(0.5, -0.25, 0.125),
        ball_position=(1.5, 0.75, 0.125),
        ball_velocity=(0.25, 0.5, -0.125),
        task_target=(7.5, 0.75, 1.5),
        steering_target=(2.0, 0.25),
        baseline_command=(0.25, 0.5, 0.125),
        previous_command=(0.125, -0.25, -0.125),
        neighbors=(("blue.defender", 3.0, -0.5), ("red.playmaker", 2.0, 0.5)),
    )


def project(obs: NavigationObservation, turn: bool = True) -> NavigationObservation:
    return canonical_navigation_observation(obs, half_turn=turn, translation_xy_m=(6.0, 0.0))


def test_off_preserves_values_without_returning_live_object() -> None:
    obs = observation()
    assert project(obs, False) == obs
    assert project(obs, False) is not obs


def test_end_effectors_rotate_with_world_without_exchanging_anatomical_sides() -> None:
    obs = replace(
        observation(),
        effector_positions=(("left_foot", 1.0, 0.125, 0.05), ("right_foot", 1.0, -0.125, 0.06)),
        committed_receiver=True,
    )
    result = project(obs)
    assert result.effector_positions == (
        ("left_foot", 5.0, -0.125, 0.05),
        ("right_foot", 5.0, 0.125, 0.06),
    )
    assert result.committed_receiver
    assert project(result).effector_positions == obs.effector_positions
    assert np.array_equal(
        local_navigation_features(obs),
        local_navigation_features(replace(obs, effector_positions=())),
    )


def test_projection_transforms_all_world_fields_not_identity_clock_or_yaw_rate() -> None:
    obs = observation()
    before = asdict(obs)
    result = project(obs)
    assert result.body_pose == (5.0, -0.5, 0.75, 0.0, 0.0, 0.0, 1.0)
    assert result.body_velocity == (-0.5, 0.25, 0.125)
    assert result.ball_position == (4.5, -0.75, 0.125)
    assert result.ball_velocity == (-0.25, -0.5, -0.125)
    assert result.task_target == (-1.5, -0.75, 1.5)
    assert result.steering_target == (4.0, -0.25)
    assert result.baseline_command == (-0.25, -0.5, 0.125)
    assert result.previous_command == (-0.125, 0.25, -0.125)
    assert result.neighbors == (("blue.defender", 3.0, 0.5), ("red.playmaker", 4.0, -0.5))
    assert (result.agent_id, result.role, result.intent, result.frame, result.time_sec) == (
        obs.agent_id,
        obs.role,
        obs.intent,
        obs.frame,
        obs.time_sec,
    )
    assert asdict(obs) == before


def test_double_rotation_restores_features_and_quaternion_orientation() -> None:
    obs = observation()
    twice = project(project(obs))
    assert twice.body_pose[:3] == obs.body_pose[:3]
    assert twice.body_pose[3:] == tuple(-x for x in obs.body_pose[3:])
    assert np.array_equal(local_navigation_features(twice), local_navigation_features(obs))


def test_no_hidden_pitch_center() -> None:
    result = canonical_navigation_observation(
        observation(), half_turn=True, translation_xy_m=(10.0, 2.0)
    )
    assert result.body_pose[:3] == (9.0, 1.5, 0.75)


@pytest.mark.parametrize("turn", [None, 0, 1, "true", np.bool_(True)])
def test_half_turn_is_explicit_boolean(turn: object) -> None:
    with pytest.raises(ValueError):
        canonical_navigation_observation(observation(), half_turn=turn, translation_xy_m=(6.0, 0.0))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "translation",
    [None, [6.0, 0.0], (0.0,), (0.0, math.nan), (0.0, math.inf), (201.0, 0.0), (True, 0.0)],
)
def test_bad_translation_rejected_even_when_off(translation: object) -> None:
    with pytest.raises(ValueError):
        canonical_navigation_observation(
            observation(), half_turn=False, translation_xy_m=translation
        )  # type: ignore[arg-type]


def test_tampered_observation_is_revalidated() -> None:
    obs = observation()
    object.__setattr__(obs, "ball_position", (math.nan, 0.0, 0.0))
    with pytest.raises(ValueError):
        project(obs)


def test_projected_world_coordinates_still_obey_navigation_bounds() -> None:
    obs = replace(observation(), ball_position=(-9999.0, 0.0, 0.1))
    with pytest.raises(ValueError):
        project(obs)


def test_world_delta_rotation_preserves_limits_and_clock() -> None:
    delta = NavigationDelta("red.finisher", 20, 0.4, (0.15, -0.2, 0.4))
    result = canonical_navigation_delta(delta, half_turn=True)
    assert result == NavigationDelta("red.finisher", 20, 0.4, (-0.15, 0.2, 0.4))
    assert canonical_navigation_delta(result, half_turn=True) == delta
    assert canonical_navigation_delta(delta, half_turn=False) == delta
    assert canonical_navigation_delta(delta, half_turn=False) is not delta


@pytest.mark.parametrize("turn", [None, 0, "true"])
def test_delta_requires_explicit_boolean(turn: object) -> None:
    with pytest.raises(ValueError):
        canonical_navigation_delta(
            NavigationDelta("red.finisher", 0, 0.0, (0.0, 0.0, 0.0)), half_turn=turn
        )  # type: ignore[arg-type]


def test_tampered_delta_cannot_increase_motion_authority() -> None:
    delta = NavigationDelta("red.finisher", 0, 0.0, (0.0, 0.0, 0.0))
    object.__setattr__(delta, "velocity_delta", (1.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        canonical_navigation_delta(delta, half_turn=True)
