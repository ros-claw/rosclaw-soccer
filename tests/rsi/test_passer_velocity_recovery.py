from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.passer_velocity_recovery import PasserVelocityRecovery


class Layout:
    def canonical(self, data):
        return np.zeros(43), np.zeros(41)

    def angular_velocity_body(self, data):
        return np.zeros(3)


class Stream:
    def __init__(self):
        self.next_frame = self.neural_calls = 0
        self.last_action = np.zeros(29, np.float32)
        self.params = SimpleNamespace(action_offset=np.zeros(29), action_scale=np.ones(29))

    def propose(self, q, v, angular, command, *, frame):
        assert frame == self.next_frame
        self.next_frame += 1
        self.neural_calls += 1
        self.last_action = np.full(29, 0.3, np.float32)
        return np.full(29, 0.3, np.float64)


class World:
    def __init__(self):
        self.layouts = {"a/": Layout()}
        self.control_steps = 0
        self.failed = False
        self.data = object()
        self.actual = np.zeros(29)

    def last_applied_target(self, prefix):
        assert prefix == "a/"
        return self.actual.copy()


def test_shadow_remembers_actual_target_then_activates_without_reset():
    world, stream = World(), Stream()
    recovery = PasserVelocityRecovery(world, "a/", stream)
    candidate = recovery.observe(np.zeros(3, np.float32))
    world.actual = np.full(29, 0.2)
    world.control_steps += 1
    row = recovery.commit()
    assert not row["active"] and stream.neural_calls == 1
    np.testing.assert_allclose(stream.last_action, 0.2)
    assert not np.array_equal(candidate, world.actual)
    recovery.activate(motion_complete=True, passer_foot_seen=True)
    candidate = recovery.observe(np.zeros(3, np.float32))
    world.actual = candidate.copy()
    world.control_steps += 1
    assert recovery.commit()["active"]
    np.testing.assert_allclose(stream.last_action, 0.3)
    assert stream.next_frame == 2


def test_no_early_or_unverified_handoff_or_skipped_physics():
    world = World()
    recovery = PasserVelocityRecovery(world, "a/", Stream())
    with pytest.raises(ValueError):
        recovery.activate(motion_complete=True, passer_foot_seen=False)
    recovery.observe(np.zeros(3, np.float32))
    with pytest.raises(ValueError):
        recovery.activate(motion_complete=True, passer_foot_seen=True)
    with pytest.raises(ValueError):
        recovery.commit()


def test_active_proposal_must_be_the_real_applied_target():
    world = World()
    recovery = PasserVelocityRecovery(world, "a/", Stream())
    recovery.activate(motion_complete=True, passer_foot_seen=True)
    recovery.observe(np.zeros(3, np.float32))
    world.control_steps += 1
    with pytest.raises(ValueError):
        recovery.commit()


def test_shadow_and_handoff_follow_one_real_shared_simulation_clock():
    import mujoco
    from test_shared_native_motor_world import world as fixture_world

    world, stream = fixture_world(), Stream()
    recovery = PasserVelocityRecovery(world, "a/", stream)
    mujoco.mj_forward(world.model, world.data)
    before = world.data.qpos.copy(), world.data.qvel.copy(), world.data.ctrl.copy()
    recovery.observe(np.zeros(3, np.float32))
    for actual, saved in zip(
        (world.data.qpos, world.data.qvel, world.data.ctrl), before, strict=True
    ):
        np.testing.assert_array_equal(actual, saved)
    world.advance({"a/": np.full(29, 0.02), "b/": np.zeros(29)})
    recovery.commit()
    np.testing.assert_allclose(stream.last_action, 0.02)
    recovery.activate(motion_complete=True, passer_foot_seen=True)
    mujoco.mj_forward(world.model, world.data)
    target = recovery.observe(np.zeros(3, np.float32))
    world.advance({"a/": target, "b/": np.zeros(29)})
    assert recovery.commit()["active"]
    assert world.native_steps == 20 and stream.neural_calls == 2
