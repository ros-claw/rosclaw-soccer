from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_cpu_batch import IndependentSonicCpuBatch
from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController


class Teacher(G1SonicRunupController):
    def __init__(self):
        self._history = deque(maxlen=10)
        self.action = np.zeros(29)
        self.config = SimpleNamespace(execution_frames=3)
        self.calls = 0
        self.fail = False

    def reset(self, state):
        self._history.clear()
        self.action = np.zeros(29)

    def observe(self, state):
        self._history.append(state.qpos.copy())

    def update(self, state, frame):
        self.calls += 1
        if self.fail:
            raise RuntimeError("injected worker failure")
        self.action = state.qpos[7:36] + frame + len(self._history)
        state.qpos[:] = 0  # Only a private snapshot, never caller physics state.
        return self.action.copy()


def states(count=2):
    q, v = np.zeros((count, 43)), np.zeros((count, 41))
    q[:, 3] = 1
    q[:, 7] = np.arange(count)
    return q, v


def test_parallel_order_private_inputs_and_serial_equivalence():
    q, v = states()
    before = q.copy()
    with (
        IndependentSonicCpuBatch([Teacher(), Teacher()], workers=1) as serial,
        IndependentSonicCpuBatch([Teacher(), Teacher()], workers=2) as parallel,
    ):
        serial.begin_episode(q, v)
        parallel.begin_episode(q, v)
        for frame in range(3):
            a, b = serial.step(frame, q, v), parallel.step(frame, q, v)
            np.testing.assert_array_equal(a, b)
            assert b[1, 0] - b[0, 0] == 1
            assert not b.flags.writeable
    np.testing.assert_array_equal(q, before)


@pytest.mark.parametrize("frame", [False, -1, 1, 3])
def test_bad_tick_fault_latches_until_whole_episode_reset(frame):
    q, v = states()
    teachers = [Teacher(), Teacher()]
    with IndependentSonicCpuBatch(teachers) as batch:
        batch.begin_episode(q, v)
        with pytest.raises(ValueError):
            batch.step(frame, q, v)
        assert all(t.calls == 0 for t in teachers)
        with pytest.raises(RuntimeError):
            batch.step(0, q, v)
        batch.begin_episode(q, v)
        batch.step(0, q, v)
        with pytest.raises(ValueError):
            batch.step(0, q, v)


@pytest.mark.parametrize("problem", ["nan", "shape", "quaternion", "dtype"])
def test_entire_input_validated_before_any_history_changes(problem):
    q, v = states()
    teachers = [Teacher(), Teacher()]
    with IndependentSonicCpuBatch(teachers) as batch:
        batch.begin_episode(q, v)
        batch.step(0, q, v)
        if problem == "nan":
            q[1, 7] = np.nan
        elif problem == "shape":
            q = q[:1]
        elif problem == "quaternion":
            q[1, 3] = 0
        else:
            q = q.astype(int)
        with pytest.raises(ValueError):
            batch.step(1, q, v)
        assert all(len(t._history) == 0 for t in teachers)


def test_all_workers_finish_before_partial_failure_returns():
    q, v = states()
    teachers = [Teacher(), Teacher()]
    teachers[0].fail = True
    with IndependentSonicCpuBatch(teachers, workers=1) as batch:
        batch.begin_episode(q, v)
        with pytest.raises(RuntimeError, match="injected"):
            batch.step(0, q, v)
        assert [t.calls for t in teachers] == [1, 1]
        with pytest.raises(RuntimeError):
            batch.step(0, q, v)


def test_submission_failure_drains_already_accepted_tasks(monkeypatch):
    q, v = states()
    teachers = [Teacher(), Teacher()]
    with IndependentSonicCpuBatch(teachers, workers=1) as batch:
        batch.begin_episode(q, v)
        original = batch._pool.submit
        calls = 0

        def submit(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("submission failed")
            return original(*args)

        monkeypatch.setattr(batch._pool, "submit", submit)
        with pytest.raises(RuntimeError, match="submission failed"):
            batch.step(0, q, v)
        assert [t.calls for t in teachers] == [1, 0]
        with pytest.raises(RuntimeError):
            batch.step(0, q, v)


@pytest.mark.parametrize("alias", ["controller", "history", "action"])
def test_shared_mutable_ownership_rejected(alias):
    teachers = [Teacher(), Teacher()]
    if alias == "controller":
        teachers[1] = teachers[0]
    elif alias == "history":
        teachers[1]._history = teachers[0]._history
    else:
        teachers[1].action = teachers[0].action.view()
    with pytest.raises(ValueError, match="independent"):
        IndependentSonicCpuBatch(teachers)


@pytest.mark.parametrize("workers", [True, 0, 9, 1.5])
def test_worker_budget_validated(workers):
    with pytest.raises(ValueError):
        IndependentSonicCpuBatch([Teacher()], workers=workers)


def test_episode_required_and_close_is_terminal():
    q, v = states()
    batch = IndependentSonicCpuBatch([Teacher(), Teacher()])
    with pytest.raises(RuntimeError):
        batch.step(0, q, v)
    batch.begin_episode(q, v)
    batch.close()
    batch.close()
    with pytest.raises(RuntimeError):
        batch.begin_episode(q, v)
    with pytest.raises(RuntimeError):
        batch.step(0, q, v)
