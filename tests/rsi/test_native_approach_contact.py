"""Synthetic lifecycle/clock tests, not native physics qualification."""

from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.native_approach_contact import (
    NativeApproachContactCourse,
    NativeApproachRecoveryTail,
)
from rosclaw_soccer.rsi.native_contact_course import NativeContactCourse


def course(monkeypatch, *, failure=False):
    env = NativeApproachContactCourse.__new__(NativeApproachContactCourse)
    env.worlds = 2
    env.layout = SimpleNamespace(rq=7)
    env.elapsed = np.zeros(2, dtype=np.int64)
    env.ended = np.zeros(2, dtype=bool)
    env.first_contact_control = np.full(2, -1, dtype=np.int64)
    env.head_failed = np.zeros(2, dtype=bool)
    env.reference_horizon = np.array([20, 20])
    env.faulted = False

    def step(self, action):
        valid = ~self.ended.copy()
        self.elapsed[valid] += 1
        q = np.zeros((2, 10, 43))
        q[..., 9], q[..., 10] = 0.8, 1.0
        if failure:
            q[1, 2, 9] = 0.64  # Transient failure even though final sample is stable.
        forces = np.zeros((2, 10, 3))
        if self.elapsed[0] == 1:
            forces[0, 3, 0] = 11.0
        return dict(
            valid=valid,
            qpos=q,
            forces=forces,
            terminated=np.zeros(2, dtype=bool),
            truncated=(self.elapsed == self.reference_horizon) & valid,
            reward=np.zeros(2),
        )

    monkeypatch.setattr(NativeContactCourse, "step", step)
    return env


def test_contact_plus14_and_no_contact_reference_end(monkeypatch):
    env = course(monkeypatch)
    for completed in range(1, 21):
        info = env.step(np.zeros((2, 29), dtype=np.float32))
        assert "reward" not in info and "legacy24_reward_diagnostic" in info
        assert bool(info["truncated"][0]) == (completed == 15)
        assert bool(info["truncated"][1]) == (completed == 20)
    assert np.array_equal(env.elapsed, [15, 20])
    assert np.array_equal(env.first_contact_control, [0, -1])
    assert not env.head_failed.any()


def test_guard_uses_all_substeps_and_failed_world_does_not_continue(monkeypatch):
    env = course(monkeypatch, failure=True)
    info = env.step(np.zeros((2, 29), dtype=np.float32))
    assert np.array_equal(info["terminated"], [False, True])
    assert np.array_equal(env.head_failed, [False, True])
    assert not info["truncated"].any()
    env.step(np.zeros((2, 29), dtype=np.float32))
    assert env.elapsed[1] == 1


def test_actual_source_end_can_precede_contact_plus14(monkeypatch):
    env = course(monkeypatch)
    env.reference_horizon[:] = 10
    for _ in range(10):
        info = env.step(np.zeros((2, 29), dtype=np.float32))
    assert info["truncated"].all()
    assert np.array_equal(env.elapsed, [10, 10])


def test_tail_requires_new_actual_completed_clock_not_old24_or_arbitrary_facade(monkeypatch):
    with pytest.raises(ValueError, match="new approach"):
        NativeApproachRecoveryTail(SimpleNamespace())
    env = course(monkeypatch)
    env.batch, env.model, env.params = object(), object(), object()
    env._closed = False
    env.phase = np.array([0.2, 0.3], dtype=np.float32)
    env.last_action = np.ones((2, 29), dtype=np.float32)
    env._observation = np.ones((2, 177), dtype=np.float32)
    env.ended[:] = True
    env.elapsed[:] = [15, 20]
    env.first_contact_control[:] = [0, -1]
    seen = []
    monkeypatch.setattr(
        NativeApproachRecoveryTail, "_prepare", lambda self, ids: seen.append(ids.copy())
    )
    tail = NativeApproachRecoveryTail(env)
    assert tail.batch is env.batch
    assert np.array_equal(seen[0], [0, 1])
    assert np.array_equal(tail.phase, env.phase)
    assert np.array_equal(tail.last_action, env.last_action)
    env.elapsed[0] = 14
    with pytest.raises(ValueError, match="bypass"):
        NativeApproachRecoveryTail(env)
