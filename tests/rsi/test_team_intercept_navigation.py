"""The trainable stance adapter stays within the guarded SIM navigation lane."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.rsi.team_intercept_navigation import TeamInterceptNavigation
from rosclaw_soccer.rsi.team_phase_intercept_navigation import TeamPhaseInterceptNavigation
from rosclaw_soccer.skills.team.navigation_option import NavigationObservation, NavigationSlot


def _observation(frame: int, *, ball_y: float = 0.1) -> NavigationObservation:
    return NavigationObservation(
        agent_id="red.playmaker",
        frame=frame,
        time_sec=frame * 0.02,
        role="playmaker",
        intent="pass",
        body_pose=(0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0),
        body_velocity=(0.0, 0.0, 0.0),
        ball_position=(0.8, ball_y, 0.115),
        ball_velocity=(-0.5, 0.0, 0.0),
        task_target=(2.0, 0.0, 0.0),
        steering_target=(1.0, 0.0),
        baseline_command=(0.2, 0.0, 0.0),
        previous_command=(0.0, 0.0, 0.0),
        neighbors=(),
        effector_positions=(("left_foot", 0.0, 0.0, 0.1), ("right_foot", 0.0, -0.2, 0.1)),
    )


def test_intercept_navigation_is_bounded_and_content_bound() -> None:
    policy = TeamInterceptNavigation(
        "red.playmaker", "sha256:" + "1" * 64, "sha256:" + "2" * 64, 0.4, 0.8
    )
    slot = NavigationSlot(policy)
    for frame in range(30):
        assert slot.propose(_observation(frame)) == (0.0, 0.0, 0.0)
    delta = slot.propose(_observation(30))
    assert delta[0] > 0 and delta[1] > 0 and delta[2] == 0.0
    assert sum(value * value for value in delta[:2]) <= 0.25**2
    assert not slot.faulted
    assert policy.history[-1][0] == 30
    with pytest.raises(ValueError):
        policy.propose(replace(_observation(31), agent_id="blue.playmaker"))


def test_uncommitted_gain_is_rejected() -> None:
    with pytest.raises(ValueError):
        TeamInterceptNavigation(
            "red.playmaker", "sha256:" + "1" * 64, "sha256:" + "2" * 64, 5.0, 0.8
        )


def test_phase_aware_navigation_uses_measured_reachable_foot() -> None:
    left_only = TeamInterceptNavigation(
        "red.playmaker", "sha256:" + "1" * 64, "sha256:" + "2" * 64, 0.0, 0.8
    )
    phase = TeamPhaseInterceptNavigation(
        "red.playmaker", "sha256:" + "1" * 64, "sha256:" + "2" * 64, 0.0, 0.8
    )
    observation = replace(
        _observation(30, ball_y=-0.15),
        effector_positions=(("left_foot", 0.0, 0.2, 0.1), ("right_foot", 0.0, -0.2, 0.2)),
    )
    assert left_only.propose(observation).velocity_delta[1] < 0
    assert phase.propose(observation).velocity_delta[1] > 0
    assert phase.contract_hash != left_only.contract_hash
