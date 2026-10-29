from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi import cftrack_shared_handoff as handoff
from rosclaw_soccer.rsi.shared_native_motor_world import NativePDParameters


class Policy:
    def __init__(self):
        self.lab2mj = np.arange(29)
        self.kps_lab = np.ones(29) * 2
        self.kds_lab = np.ones(29) * 3
        self.state_cmd = object()
        self.policy_output = object()


class World:
    def __init__(self):
        self.parameters = {"b/": NativePDParameters(np.ones(29), np.ones(29), np.ones(29))}
        self._private_policy_owners = {}
        self._torque_layer_owners = {}
        self.layouts = {"b/": SimpleNamespace(ball=0)}
        self.data = SimpleNamespace(
            xpos=np.zeros((1, 3)), xquat=np.array([[1.0, 0, 0, 0]]), time=1.0
        )
        self.control_steps = 50
        self.switches = 0

    def _require_current_boundary(self):
        pass

    def switch_pd_profile(self, prefix, params):
        self.parameters[prefix] = params.owned_copy()
        self.switches += 1
        return {"before": {}, "after": {}, "physics_unchanged": True}

    def last_applied_target(self, prefix):
        return np.zeros(29)


class History:
    def __init__(self, world):
        self.world = world

    def belongs_to(self, world, prefix):
        return self.world is world and prefix == "b/"

    def at_current_boundary(self):
        return (None,) * 5


class Player:
    def __init__(self, world, prefix, policy, **kwargs):
        world._private_policy_owners[prefix] = (
            id(policy),
            id(policy.state_cmd),
            id(policy.policy_output),
        )


def fixture(monkeypatch):
    monkeypatch.setattr(handoff, "SharedNativeMotorWorld", World)
    monkeypatch.setattr(handoff, "CFtrackLiveHistory", History)
    monkeypatch.setattr(handoff, "SharedOmniContactPlayer", Player)
    monkeypatch.setattr(handoff, "snapshot", lambda data: {"time": np.array([data.time])})
    return World()


def test_owned_handoff_preserves_clock_and_history(monkeypatch):
    world = fixture(monkeypatch)
    monkeypatch.setattr(
        handoff, "install_owned_cftrack_history", lambda player, history: {"verified": True}
    )
    player, receipt = handoff.handoff_shared_cftrack(
        world, "b/", History(world), Policy(), expected_policy_type=Policy, goal=np.zeros(3)
    )
    assert isinstance(player, Player)
    assert receipt["body_ball_clock_unchanged"] is True
    assert receipt["physics_success_proven"] is False
    assert world.switches == 1
    np.testing.assert_array_equal(world.parameters["b/"].stiffness, np.full(29, 2))


def test_failed_history_install_restores_gains_and_owned_registration(monkeypatch):
    world = fixture(monkeypatch)

    def fail(*args):
        raise ValueError("bad own history")

    monkeypatch.setattr(handoff, "install_owned_cftrack_history", fail)
    with pytest.raises(ValueError, match="bad own history"):
        handoff.handoff_shared_cftrack(
            world, "b/", History(world), Policy(), expected_policy_type=Policy, goal=np.zeros(3)
        )
    assert world.switches == 2
    assert "b/" not in world._private_policy_owners
    np.testing.assert_array_equal(world.parameters["b/"].stiffness, np.ones(29))


def test_two_body_native_clock_advances_after_owned_history_switch(monkeypatch):
    import copy

    from test_cftrack_measured_history_install import ScratchPolicy
    from test_shared_robonaldo_player import setup

    from rosclaw_soccer.rsi.cftrack_live_history import CFtrackLiveHistory

    world = setup()
    history = CFtrackLiveHistory(world, "b/")
    for i in range(5):
        world.advance({"a/": np.zeros(29), "b/": np.full(29, (i + 1) * 0.001)})
        history.capture()
    monkeypatch.setattr(
        "rosclaw_soccer.rsi.cftrack_measured_history_install.copy_cftrack_policy", copy.deepcopy
    )
    monkeypatch.setattr(
        "rosclaw_soccer.rsi.cftrack_measured_history_install.cftrack_state_hash",
        lambda policy: "same",
    )
    monkeypatch.setattr(
        "rosclaw_soccer.rsi.cftrack_measured_history_install.assemble_current_ball_observation",
        lambda policy: np.zeros(141, np.float32),
    )
    before_q = world.data.qpos.copy()
    before_v = world.data.qvel.copy()
    before_time = world.data.time
    player, receipt = handoff.handoff_shared_cftrack(
        world,
        "b/",
        history,
        ScratchPolicy(),
        expected_policy_type=ScratchPolicy,
        goal=np.array([5.0, 0, 0.11]),
    )
    np.testing.assert_array_equal(world.data.qpos, before_q)
    np.testing.assert_array_equal(world.data.qvel, before_v)
    assert world.data.time == before_time
    assert receipt["measured_history"]["sample_controls"] == [1, 2, 3, 4, 5]
    target, _ = player.propose()
    world.advance({"a/": np.zeros(29), "b/": target})
    player.commit()
    assert world.control_steps == 6 and player.commits == 1
