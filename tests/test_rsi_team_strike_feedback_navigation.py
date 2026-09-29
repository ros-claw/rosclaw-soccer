from dataclasses import replace

import pytest

from rosclaw_soccer.rsi.team_strike_feedback_navigation import TeamStrikeFeedbackNavigation
from rosclaw_soccer.skills.team.navigation_option import NavigationObservation


def _observation(*, intent="shoot", ball=(1.0, 0.0, 0.115), target=(7.0, 0.0, 0.0)):
    return NavigationObservation(
        agent_id="red.finisher",
        frame=50,
        time_sec=1.0,
        role="finisher",
        intent=intent,
        body_pose=(0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0),
        body_velocity=(0.0, 0.0, 0.0),
        ball_position=ball,
        ball_velocity=(0.0, 0.0, 0.0),
        task_target=target,
        steering_target=(0.0, 0.0),
        baseline_command=(0.0, 0.0, 0.0),
        previous_command=(0.0, 0.0, 0.0),
        neighbors=(),
    )


def _policy(**overrides):
    payload = dict(
        agent_id="red.finisher",
        foundation_hash="sha256:" + "a" * 64,
        foundation_config_hash="sha256:" + "b" * 64,
        position_gain=1.2,
        prediction_horizon_sec=0.6,
        stance_depth_m=0.36,
        stance_lateral_m=-0.19,
        body_velocity_damping=0.2,
    )
    payload.update(overrides)
    return TeamStrikeFeedbackNavigation(**payload)


def test_strike_feedback_is_bounded_task_scoped_and_rotation_equivariant():
    policy = _policy()
    result = policy.propose(_observation())
    assert result is not None
    assert result.velocity_delta[0] > 0
    assert abs(result.velocity_delta[1]) > 0
    assert sum(x * x for x in result.velocity_delta[:2]) <= 0.25**2 + 1e-9
    assert policy.propose(_observation(intent="receive")) is None
    rotated = _policy().propose(_observation(ball=(0.0, 1.0, 0.115), target=(0.0, 7.0, 0.0)))
    assert rotated is not None
    assert rotated.velocity_delta[0] == pytest.approx(-result.velocity_delta[1])
    assert rotated.velocity_delta[1] == pytest.approx(result.velocity_delta[0])


def test_strike_feedback_parameters_are_hash_bound_and_finite():
    first = _policy()
    assert _policy(position_gain=0.8).contract_hash != first.contract_hash
    for invalid in (-0.1, 1.6, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            _policy(position_gain=invalid)
    with pytest.raises(ValueError, match="foreign"):
        first.propose(replace(_observation(), agent_id="blue.finisher"))
