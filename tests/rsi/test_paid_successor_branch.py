from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
from rosclaw_soccer.rsi.paid_successor_branch import begin_successor_branch


def fixture(monkeypatch, *, corrupt=False):
    """Flow-only fixture, no G1/native success implied."""
    import mujoco

    data = SimpleNamespace(
        **{
            key: np.zeros(shape)
            for key, shape in (
                ("qpos", 43),
                ("qvel", 41),
                ("ctrl", 29),
                ("qacc_warmstart", 41),
                ("mocap_pos", (1, 3)),
                ("mocap_quat", (1, 4)),
                ("qfrc_applied", 41),
                ("xfrc_applied", (33, 6)),
                ("act", 0),
            )
        },
        time=0.0,
    )
    entry = snapshot(data)
    entry["qpos"][:] = np.arange(43) * 0.01
    entry["time"] = np.array(12.0)
    entry["qacc_warmstart"][:] = 0.2
    calls = []
    monkeypatch.setattr(mujoco, "mj_forward", lambda m, d: calls.append("forward"))
    ep = SimpleNamespace(
        env=SimpleNamespace(data=data, model=None, goal_mocap_id=0), clip=None, frame=100, done=True
    )

    def begin(clip, *, goal):
        assert calls == ["forward"]
        assert goal.shape == (3,)
        ep.frame, ep.done = 0, False
        if corrupt:
            data.qpos[0] = 1

    ep.begin_current_state = begin
    bench = SimpleNamespace(make=lambda case: ep)
    return bench, entry, np.ones(29, np.float32) * 0.1


def test_training_branch_keeps_recorded_clock_full_integration_and_previous_action(monkeypatch):
    bench, entry, previous = fixture(monkeypatch)
    ep = begin_successor_branch(bench, {}, entry, previous)
    assert ep.env.data.time == 12 and np.array_equal(ep.last_action, previous)
    assert np.array_equal(ep.env.data.qacc_warmstart, entry["qacc_warmstart"])


def test_branch_cannot_hide_robot_reset_by_its_next_skill_boundary(monkeypatch):
    bench, entry, previous = fixture(monkeypatch, corrupt=True)
    with pytest.raises(ValueError):
        begin_successor_branch(bench, {}, entry, previous)


def test_branch_refuses_incomplete_entry_and_nonactual_action(monkeypatch):
    bench, entry, previous = fixture(monkeypatch)
    with pytest.raises(ValueError):
        begin_successor_branch(bench, {}, entry, previous.astype(np.float64))
    del entry["qacc_warmstart"]
    with pytest.raises(ValueError):
        begin_successor_branch(bench, {}, entry, previous)
