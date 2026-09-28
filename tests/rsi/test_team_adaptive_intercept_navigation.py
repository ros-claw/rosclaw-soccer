"""Physical-search interception remains measured, bounded and SIM_ONLY."""

from __future__ import annotations

import math

import pytest

from rosclaw_soccer.rsi.team_adaptive_intercept_navigation import TeamAdaptiveInterceptNavigation
from rosclaw_soccer.skills.team.navigation_option import NavigationObservation


def _observation(frame: int) -> NavigationObservation:
    return NavigationObservation(
        agent_id="red.playmaker",
        frame=frame,
        time_sec=frame * 0.02,
        role="playmaker",
        intent="pass",
        body_pose=(0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0),
        body_velocity=(0.0, 0.0, 0.0),
        ball_position=(1.4, -0.2, 0.115),
        ball_velocity=(-0.4, 0.0, 0.0),
        task_target=(1.0, 0.0, 0.0),
        steering_target=(1.0, 0.0),
        baseline_command=(0.0, 0.0, 0.0),
        previous_command=(0.0, 0.0, 0.0),
        neighbors=(),
        effector_positions=(
            ("left_foot", 0.0, 0.0, 0.2),
            ("right_foot", 0.0, -0.2, 0.1),
        ),
    )


def test_adaptive_interception_is_causal_bounded_and_phase_selected() -> None:
    policy = TeamAdaptiveInterceptNavigation(
        agent_id="red.playmaker",
        foundation_hash="sha256:" + "1" * 64,
        foundation_config_hash="sha256:" + "2" * 64,
        forward_gain=0.8,
        lateral_gain=1.6,
        target_gap_m=0.3,
        activation_max_gap_m=2.0,
        speed_cap_mps=0.25,
    )
    assert policy.propose(_observation(29)).velocity_delta == (0.0, 0.0, 0.0)
    velocity = policy.propose(_observation(30)).velocity_delta
    assert velocity[0] > 0 and velocity[1] < 0
    assert math.hypot(*velocity[:2]) <= 0.250000001
    assert policy.activation_ceiling == "SIM_ONLY"


def test_adaptive_interception_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        TeamAdaptiveInterceptNavigation(
            agent_id="red.playmaker",
            foundation_hash="sha256:" + "1" * 64,
            foundation_config_hash="sha256:" + "2" * 64,
            forward_gain=0.8,
            lateral_gain=float("nan"),
            target_gap_m=0.3,
            activation_max_gap_m=2.0,
            speed_cap_mps=0.25,
        )
