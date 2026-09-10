from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.skills.team.motor_option import (
    TeamBallContact,
    TeamMotorPhysicsObservation,
    TeamMotorTarget,
)


def observation():
    return TeamMotorPhysicsObservation(
        0.002,
        (0.0,) * 43,
        (0.0,) * 41,
        True,
        2.0,
        3.0,
        observer_agent_id="red.a",
        contacts_complete=True,
        ball_contacts=(
            TeamBallContact(5, "red.a", "left_foot", 2.0),
            TeamBallContact(7, "blue.b", "right_foot", 3.0),
        ),
    )


def test_other_players_foot_remains_in_legacy_other_force():
    value = observation()
    assert value.ball_contacts[1].is_foot
    assert value.other_non_ground_normal_force_n == 3.0
    with pytest.raises(FrozenInstanceError):
        value.ball_contacts[1].normal_force_n = 0
    with pytest.raises(ValueError):
        replace(value, other_non_ground_normal_force_n=0.0)


def test_legacy_or_explicitly_complete_empty_records():
    old = TeamMotorPhysicsObservation(0.002, (0.0,) * 43, (0.0,) * 41, True, 2.0, 3.0)
    assert not old.contacts_complete and not old.ball_contacts
    empty = replace(
        old,
        foot_normal_force_n=0,
        other_non_ground_normal_force_n=0,
        observer_agent_id="red.a",
        contacts_complete=True,
    )
    assert empty.contacts_complete and empty.ball_contacts == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"geometry_id": True},
        {"geometry_id": -1},
        {"agent_id": None},
        {"agent_id": 1},
        {"agent_id": "../peer"},
        {"effector": "environment"},
        {"effector": "unknown"},
        {"effector": []},
        {"normal_force_n": True},
        {"normal_force_n": float("nan")},
        {"normal_force_n": -1},
    ],
)
def test_invalid_counterpart_rejected(changes):
    with pytest.raises(ValueError):
        replace(TeamBallContact(5, "red.a", "left_foot", 2.0), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"contacts_complete": 1},
        {"contacts_complete": False},
        {"observer_agent_id": None},
        {"observer_agent_id": "blue.b"},
        {"ball_contacts": []},
        {"ball_contacts": ("foot",)},
        {"ball_contacts": ()},
        {"foot_normal_force_n": 2.000001},
    ],
)
def test_partial_or_inconsistent_attribution_rejected(changes):
    with pytest.raises(ValueError):
        replace(observation(), **changes)


def test_world_attributes_each_counterpart_without_mutating_physics(monkeypatch):
    import mujoco

    from rosclaw_soccer.skills.team.independent_team_world import _observe_team_motor_physics

    class Observer:
        def __init__(self):
            self.seen = []

        def observe_physics(self, value):
            self.seen.append(value)

    q = np.zeros(43)
    q[2] = 0.75
    q[3] = q[39] = 1
    data = SimpleNamespace(
        qpos=q,
        qvel=np.zeros(41),
        time=0.002,
        ncon=5,
        contact=[SimpleNamespace(geom1=9, geom2=g) for g in (5, 7, 8, 11, 0)],
    )
    model = SimpleNamespace(
        jnt_range=np.tile([-10.0, 10.0], (29, 1)),
        jnt_limited=np.ones(29),
        geom=lambda g: SimpleNamespace(name="floor" if g == 0 else "body"),
    )

    def player(agent, left, right, geoms):
        return SimpleNamespace(
            cell=SimpleNamespace(agent_id=agent),
            qpos_base=0,
            qvel_base=0,
            joint_qpos=np.arange(7, 36),
            joint_qvel=np.arange(6, 35),
            joint_ids=np.arange(29),
            left_foot_geoms={left},
            right_foot_geoms={right},
            robot_geoms=geoms,
        )

    def force(model, data, index, wrench):
        wrench[0] = (2.0, 3.0, 4.0, 5.0, 100.0)[index]

    monkeypatch.setattr(mujoco, "mj_contactForce", force)
    observer = Observer()
    targets = {"red.a": TeamMotorTarget((0.0,) * 29, (50.0,) * 29, (1.0,) * 29)}
    faults = set()
    before = q.copy()
    _observe_team_motor_physics(
        model,
        data,
        (player("red.a", 5, 6, {5, 6}), player("blue.b", 7, 10, {7, 8, 10})),
        {"red.a": observer},
        targets,
        faults,
        9,
        36,
        35,
    )
    np.testing.assert_array_equal(q, before)
    assert targets and not faults
    value = observer.seen[0]
    assert value.contacts_complete and value.observer_agent_id == "red.a"
    assert value.foot_normal_force_n == 2 and value.other_non_ground_normal_force_n == 5
    assert [(c.agent_id, c.effector) for c in value.ball_contacts] == [
        ("red.a", "left_foot"),
        ("blue.b", "left_foot"),
        ("blue.b", "body"),
        (None, "environment"),
    ]
