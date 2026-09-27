"""Cheap policy-boundary checks for the SIM_ONLY contact optimizer."""

import math

import numpy as np
import pytest
from rsi_sonic_contact_residual_cem import (
    AIM_RESERVED,
    AIM_TRAIN,
    SPATIOTEMPORAL_RESERVED,
    SPATIOTEMPORAL_TRAIN,
    _params,
    _score,
    _spatiotemporal_timing,
)


def test_spatiotemporal_course_partitions_are_disjoint() -> None:
    assert len(SPATIOTEMPORAL_TRAIN) == 9
    assert len(SPATIOTEMPORAL_RESERVED) == 3
    assert not set(SPATIOTEMPORAL_TRAIN) & set(SPATIOTEMPORAL_RESERVED)
    assert len(set(SPATIOTEMPORAL_TRAIN)) == len(SPATIOTEMPORAL_TRAIN)
    assert len(AIM_TRAIN) == 12
    assert len(AIM_RESERVED) == 3
    assert not set(AIM_TRAIN) & set(AIM_RESERVED)


def test_spatiotemporal_parent_preserves_original_envelope() -> None:
    for x, y in (*SPATIOTEMPORAL_TRAIN, *SPATIOTEMPORAL_RESERVED):
        assert _spatiotemporal_timing((0.48, 0.0, 0.18, 0.0), x, y) == (0.48, 0.18)


def test_spatiotemporal_envelope_remains_bounded() -> None:
    for x, y in (*SPATIOTEMPORAL_TRAIN, *SPATIOTEMPORAL_RESERVED):
        center, sigma = _spatiotemporal_timing((0.65, 2.0, 0.25, 1.0), x, y)
        assert 0.35 <= center <= 0.65
        assert 0.08 <= sigma <= 0.25


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_candidate_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="finite four-parameter"):
        _params(np.array((0.48, bad, 0.18, 0.0)))


def test_unsafe_or_nonfoot_goal_cannot_score_as_success() -> None:
    report = {
        "minimum_pelvis_height_m": 0.7,
        "peak_pelvis_tilt_rad": 0.2,
        "actuator_saturation_fraction": 0.0,
        "first_robot_ball_contact_is_foot": True,
        "whole_ball_goal_crossed": True,
        "peak_ball_speed_mps": 3.0,
    }
    assert _score(report) == 13.0
    assert _score({**report, "minimum_pelvis_height_m": 0.61}) == -10.0
    assert _score({**report, "peak_ball_speed_mps": math.nan}) == -10.0
    assert _score({**report, "first_robot_ball_contact_is_foot": False}) == -2.0


def test_aim_score_prefers_precise_goal_to_faster_wide_goal() -> None:
    report = {
        "minimum_pelvis_height_m": 0.7,
        "peak_pelvis_tilt_rad": 0.2,
        "actuator_saturation_fraction": 0.0,
        "first_robot_ball_contact_is_foot": True,
        "whole_ball_goal_crossed": True,
        "peak_ball_speed_mps": 3.0,
        "goal_crossing_ball_center_xyz_m": [5.11, 0.05, 0.11],
    }
    accurate = _score(report, mode="aim")
    wide = _score(
        {
            **report,
            "peak_ball_speed_mps": 4.0,
            "goal_crossing_ball_center_xyz_m": [5.11, 0.6, 0.11],
        },
        mode="aim",
    )
    assert accurate > wide
    assert _score({**report, "whole_ball_goal_crossed": False}, mode="aim") == 1.0
