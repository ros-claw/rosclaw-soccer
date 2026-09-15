from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.growth.role_self_model import MatchRole, TacticalIntent
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    _movement_command,
)


def command(team, enabled, *, held=False, committed=True, duel=False, opponent=False):
    blue = team == "blue"
    root = np.array([4.25, 1.1]) if blue else np.array([1.75, -1.1])
    ball = np.array([4.92, -0.49]) if blue else np.array([1.08, 0.49])
    cell = SimpleNamespace(
        agent_id=f"{team}.playmaker",
        self_model=SimpleNamespace(
            primary_role=MatchRole.PLAYMAKER,
            team_id=team,
            teammate_ids=(),
            opponent_ids=("opponent.defender",),
        ),
    )
    controller = SimpleNamespace(
        cell=cell, qpos_base=0, last_world_command=None, left_ankle_body=0, right_ankle_body=1
    )
    quaternion = [0.0, 0.0, 0.0, 1.0] if blue else [1.0, 0.0, 0.0, 0.0]
    data = SimpleNamespace(
        qpos=np.array([*root, 0.78, *quaternion, *ball, 0.115]),
        qvel=np.zeros(3),
        xpos=np.array([[root[0], root[1] + 0.1, 0.04], [root[0], root[1] - 0.1, 0.04]]),
    )
    return _movement_command(
        controller=controller,
        decision=SimpleNamespace(intent=TacticalIntent.RECEIVE, target_position_m=(*ball, 0.0)),
        positions={cell.agent_id: root},
        data=data,
        ball_qpos=7,
        ball_qvel=0,
        possession_agent_id="opponent.defender" if opponent else None,
        committed_receiver=committed,
        active_receiver=False,
        post_receive_hold=held,
        receive_foot_lateral_offset_m=0.12,
        strike_target_position_m=None,
        config=IndependentTeamWorldConfig(
            bilateral_goals=True,
            rotation_equivariant_receive_heading=enabled,
            rotation_equivariant_duel_side=duel,
        ),
    )


def test_default_identity_is_unchanged_and_optin_is_bound():
    old = IndependentTeamWorldConfig()
    assert (
        old.config_hash == "sha256:fd442bce83f737c64e2ee59ecc09dc204376100f9476a21361b41b538d8c41d1"
    )
    new = replace(old, bilateral_goals=True, rotation_equivariant_receive_heading=True)
    assert new.config_hash != replace(new, rotation_equivariant_receive_heading=False).config_hash
    with pytest.raises(ValueError):
        replace(old, rotation_equivariant_receive_heading=True)


@pytest.mark.parametrize("value", [0, 1, None, "yes", np.bool_(True)])
def test_explicit_boolean_required(value):
    with pytest.raises(ValueError):
        IndependentTeamWorldConfig(bilateral_goals=True, rotation_equivariant_receive_heading=value)


def test_measured_initial_heading_counterexample_and_rotation_fix():
    old_blue, old_red = command("blue", False), command("red", False)
    assert old_blue[2] == pytest.approx(-0.8)
    assert old_red[2] == pytest.approx(0.47856429901227604)
    blue, red = command("blue", True), command("red", True)
    np.testing.assert_array_equal(blue, old_blue)
    assert blue[2] == red[2]
    np.testing.assert_allclose(blue[:2], -red[:2], atol=1e-12)


@pytest.mark.parametrize("team", ["red", "blue"])
def test_heading_option_cannot_override_hold_or_uncommitted_movement(team):
    np.testing.assert_array_equal(command(team, True, held=True), np.zeros(3))
    np.testing.assert_array_equal(
        command(team, True, committed=False), command(team, False, committed=False)
    )


@pytest.mark.parametrize("value", [0, 1, None, "yes", np.bool_(True)])
def test_duel_side_contract_is_explicit(value):
    with pytest.raises(ValueError):
        IndependentTeamWorldConfig(bilateral_goals=True, rotation_equivariant_duel_side=value)


def test_duel_side_optin_is_bound_and_requires_bilateral_world():
    with pytest.raises(ValueError):
        IndependentTeamWorldConfig(rotation_equivariant_duel_side=True)
    a = IndependentTeamWorldConfig(bilateral_goals=True)
    assert a.config_hash != replace(a, rotation_equivariant_duel_side=True).config_hash


def test_opponent_receive_offset_rotates_without_changing_clearance_or_legacy_red():
    old_red = command("red", False, committed=False, opponent=True)
    old_blue = command("blue", False, committed=False, opponent=True)
    red = command("red", False, committed=False, opponent=True, duel=True)
    blue = command("blue", False, committed=False, opponent=True, duel=True)
    np.testing.assert_array_equal(red, old_red)
    assert not np.allclose(old_red[:2], -old_blue[:2])
    np.testing.assert_allclose(red[:2], -blue[:2], atol=1e-12)
    for team in ("red", "blue"):
        np.testing.assert_array_equal(
            command(team, False, committed=False, opponent=False, duel=True),
            command(team, False, committed=False, opponent=False, duel=False),
        )
        np.testing.assert_array_equal(
            command(team, False, committed=False, opponent=True, duel=True, held=True),
            np.zeros(3),
        )
