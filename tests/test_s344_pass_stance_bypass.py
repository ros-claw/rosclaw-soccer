from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.growth.owned_ball_contact import OwnedBallContactPolicy
from rosclaw_soccer.growth.role_self_model import MatchRole, TacticalIntent
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    _movement_command,
)


@pytest.mark.parametrize("angle", [0.0, 0.7, np.pi, -1.4])
def test_wrong_side_path_is_rotation_and_translation_equivariant(angle):
    p = OwnedBallContactPolicy()
    current, ball, goal = np.array([-0.8, 0.0]), np.zeros(2), np.array([-3.0, 0.0])
    expected = p.approach_waypoint(
        tuple(current), tuple(ball), tuple(goal), lateral_clearance_m=0.55
    )
    r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    offset = np.array([3.0, -2.0])
    actual = p.approach_waypoint(
        *(tuple(r @ v + offset) for v in (current, ball, goal)), lateral_clearance_m=0.55
    )
    np.testing.assert_allclose(actual, r @ expected + offset, atol=1e-9)
    assert expected[0] == pytest.approx(current[0])
    assert abs(expected[1]) == pytest.approx(0.55)


def test_clear_lateral_lane_advances_around_ball_then_reaches_stance():
    p = OwnedBallContactPolicy()
    target = p.approach_waypoint((-0.8, 0.6), (0.0, 0.0), (-3.0, 0.0), lateral_clearance_m=0.55)
    assert target == pytest.approx((0.44, 0.6))
    final = p.approach_waypoint((0.2, 0.6), (0.0, 0.0), (-3.0, 0.0), lateral_clearance_m=0.55)
    assert final == pytest.approx(p.stance((0.0, 0.0), (-3.0, 0.0))[0])
    # The final stance itself must not trigger another bypass loop.
    assert p.approach_waypoint(
        final, (0.0, 0.0), (-3.0, 0.0), lateral_clearance_m=0.55
    ) == pytest.approx(final)


@pytest.mark.parametrize("radius", [True, 0.1, 1.2, float("nan"), float("inf")])
def test_invalid_clearance_rejected(radius):
    with pytest.raises(ValueError):
        OwnedBallContactPolicy().approach_waypoint(
            (0.0, 0.0), (1.0, 0.0), (4.0, 0.0), lateral_clearance_m=radius
        )


def test_config_requires_explicit_policy_and_preserves_default_hash():
    config = IndependentTeamWorldConfig(owned_contact_policy=OwnedBallContactPolicy())
    assert config.config_hash == replace(config, pass_stance_bypass=False).config_hash
    assert config.config_hash != replace(config, pass_stance_bypass=True).config_hash
    for bad in [1, "true", None]:
        with pytest.raises(ValueError):
            replace(config, pass_stance_bypass=bad)
    with pytest.raises(ValueError):
        IndependentTeamWorldConfig(pass_stance_bypass=True)


def test_world_uses_waypoint_but_keeps_velocity_guards_and_receive_hold():
    cell = SimpleNamespace(
        agent_id="red.finisher",
        self_model=SimpleNamespace(
            primary_role=MatchRole.FINISHER,
            team_id="red",
            teammate_ids=(),
            opponent_ids=(),
        ),
    )
    controller = SimpleNamespace(cell=cell, qpos_base=0, last_world_command=None)
    data = SimpleNamespace(
        qpos=np.array([-0.8, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.115]), qvel=np.zeros(3)
    )
    decision = SimpleNamespace(intent=TacticalIntent.PASS, target_position_m=(-3.0, 0.0, 0.0))
    config = IndependentTeamWorldConfig(
        owned_contact_policy=OwnedBallContactPolicy(), pass_stance_bypass=True
    )
    kwargs = dict(
        controller=controller,
        decision=decision,
        positions={cell.agent_id: np.array([-0.8, 0.0])},
        data=data,
        ball_qpos=7,
        ball_qvel=0,
        possession_agent_id=None,
        prospective_contact=True,
        committed_receiver=False,
        active_receiver=False,
        post_receive_hold=False,
        receive_foot_lateral_offset_m=0.18,
        strike_target_position_m=None,
    )
    command = _movement_command(**kwargs, config=config)
    old = _movement_command(**kwargs, config=replace(config, pass_stance_bypass=False))
    assert abs(command[0]) < 1e-10 < old[0]
    assert 0.0 < abs(command[1]) <= config.maximum_acceleration_mps2 * 0.02 + 1e-9
    assert np.linalg.norm(command[:2]) <= config.maximum_speed_mps + 1e-9
    assert abs(command[2]) <= config.maximum_yaw_rate_radps
    np.testing.assert_array_equal(
        _movement_command(**{**kwargs, "post_receive_hold": True}, config=config), np.zeros(3)
    )
