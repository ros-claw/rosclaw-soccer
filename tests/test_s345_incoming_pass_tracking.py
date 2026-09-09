from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import test_s220_basic_ball_play as basic_ball_play

from rosclaw_soccer.growth.pass_handoff import PassHandoff
from rosclaw_soccer.growth.role_self_model import MatchRole, TacticalIntent
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    _movement_command,
)


def test_flight_tracking_needs_physical_foot_contact_and_measured_approach():
    handoff = PassHandoff("red.finisher", "red.playmaker", 1.0)
    values = dict(ball_xy=(0.0, 0.0), ball_velocity_xy=(1.0, 0.0), receiver_xy=(2.0, 0.1))
    assert not handoff.can_track_incoming_ball(1.1, **values)
    launched = handoff.observe_contact(
        agent_id=handoff.source, foot=True, force_n=10.0, time_sec=1.2
    )
    assert launched.can_track_incoming_ball(1.3, **values)
    assert not launched.can_track_incoming_ball(1.1, **values)
    for velocity in [(0.0, 0.0), (-1.0, 0.0), (0.0, 1.0)]:
        assert not launched.can_track_incoming_ball(1.3, **{**values, "ball_velocity_xy": velocity})
    assert not launched.can_track_incoming_ball(4.2, **values)
    assert not replace(launched, interrupted=True).can_track_incoming_ball(1.3, **values)
    assert launched.source_foot_contact_sec == 1.2
    for vector in [(float("nan"), 0.0), (True, 0.0), (0.0,), [0.0, 0.0]]:
        with pytest.raises(ValueError):
            launched.can_track_incoming_ball(1.3, **{**values, "ball_xy": vector})


def test_receiver_can_anticipate_while_source_contact_label_is_remembered(monkeypatch):
    course = basic_ball_play.course.__wrapped__(monkeypatch)
    cell = next(c for c in course.cells if c.agent_id == "red.playmaker")
    value = basic_ball_play.observation(course, cell.agent_id, "red.finisher")
    assert cell.decide(value).intent is TacticalIntent.SUPPORT
    incoming = replace(value, active_receive_source_agent_id="red.finisher")
    assert cell.decide(incoming).intent is TacticalIntent.RECEIVE
    assert incoming.possession_agent_id == "red.finisher"
    assert (
        cell.decide(replace(incoming, possession_agent_id="blue.defender")).intent
        is not TacticalIntent.PASS
    )
    assert (
        cell.decide(replace(incoming, self_state=replace(incoming.self_state, stable=False))).intent
        is TacticalIntent.RECOVER
    )


@pytest.mark.parametrize("enabled,active", [(False, True), (True, False), (True, True)])
def test_near_ball_braking_keeps_only_causal_lateral_correction(enabled, active):
    cell = SimpleNamespace(
        agent_id="red.playmaker",
        self_model=SimpleNamespace(
            primary_role=MatchRole.PLAYMAKER,
            team_id="red",
            teammate_ids=("red.finisher",),
            opponent_ids=(),
        ),
    )
    controller = SimpleNamespace(
        cell=cell, qpos_base=0, last_world_command=None, left_ankle_body=0, right_ankle_body=1
    )
    data = SimpleNamespace(
        qpos=np.array([0.0, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0, -0.3, 0.25, 0.115]),
        qvel=np.array([1.0, 0.0, 0.0]),
        xpos=np.array([[0.0, 0.05, 0.04], [0.0, -0.1, 0.04]]),
    )
    decision = SimpleNamespace(intent=TacticalIntent.RECEIVE, target_position_m=(-0.3, 0.25, 0.0))
    config = IndependentTeamWorldConfig(
        strict_receive_handoff=True, receive_lateral_braking=enabled, arrival_radius_m=0.08
    )
    kwargs = dict(
        controller=controller,
        decision=decision,
        positions={cell.agent_id: np.zeros(2)},
        data=data,
        ball_qpos=7,
        ball_qvel=0,
        possession_agent_id="red.finisher",
        committed_receiver=True,
        active_receiver=active,
        post_receive_hold=False,
        receive_foot_lateral_offset_m=0.12,
        strike_target_position_m=None,
        config=config,
    )
    cmd = _movement_command(**kwargs)
    assert cmd[0] == pytest.approx(0.0)
    if enabled and active:
        assert cmd[1] > 0
    else:
        np.testing.assert_array_equal(cmd[:2], np.zeros(2))
    assert np.linalg.norm(cmd[:2]) <= config.maximum_acceleration_mps2 * 0.02 + 1e-9
    np.testing.assert_array_equal(
        _movement_command(**{**kwargs, "post_receive_hold": True}), np.zeros(3)
    )


def test_lateral_braking_config_is_explicit_and_does_not_change_default_identity():
    old = IndependentTeamWorldConfig()
    assert (
        old.config_hash == "sha256:fd442bce83f737c64e2ee59ecc09dc204376100f9476a21361b41b538d8c41d1"
    )
    for invalid in [1, None, "yes"]:
        with pytest.raises(ValueError):
            replace(old, receive_lateral_braking=invalid, strict_receive_handoff=True)
    with pytest.raises(ValueError):
        replace(old, receive_lateral_braking=True)
