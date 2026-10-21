from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.native_recovery_tail import NativeRecoveryTail


def course():
    return SimpleNamespace(
        protocol="native-contact24-v1",
        batch=object(),
        _closed=False,
        faulted=False,
        ended=np.array([True, True]),
        worlds=2,
        elapsed=np.array([24, 5]),
        phase=np.array([0.2, 0.4], dtype=np.float32),
        last_action=np.ones((2, 29), dtype=np.float32),
        observation=np.ones((2, 177), dtype=np.float32),
        model=object(),
        layout=object(),
        params=object(),
    )


def test_handoff_uses_same_batch_and_only_prepares_live_worlds(monkeypatch):
    c = course()
    seen = []
    monkeypatch.setattr(NativeRecoveryTail, "_prepare", lambda self, ids: seen.append(ids.copy()))
    tail = NativeRecoveryTail(c, contact_failed=np.array([False, True]))
    assert tail.batch is c.batch
    assert np.array_equal(tail.active_ids, [0])
    assert np.array_equal(seen[0], [0])
    assert np.array_equal(tail.phase, c.phase)
    assert np.array_equal(tail.last_action, c.last_action)
    # Terminal observation is not replaced by zero padding.
    assert np.array_equal(tail.observation[1], c.observation[1])
    tail.ended[:] = True
    with pytest.raises(ValueError, match="active recovery"):
        tail.step(np.zeros((2, 29), dtype=np.float32))


def test_no_handoff_before_all_contacts_end_or_after_failure(monkeypatch):
    monkeypatch.setattr(NativeRecoveryTail, "_prepare", lambda self, ids: None)
    c = course()
    for field, value in (("ended", np.array([True, False])), ("faulted", True), ("_closed", True)):
        bad = SimpleNamespace(**vars(c))
        setattr(bad, field, value)
        with pytest.raises(ValueError):
            NativeRecoveryTail(bad, contact_failed=np.array([False, True]))
    with pytest.raises(ValueError):
        NativeRecoveryTail(c, contact_failed=np.array([False, False]))
    with pytest.raises(ValueError):
        NativeRecoveryTail(c, contact_failed=np.array([0, 1]))
