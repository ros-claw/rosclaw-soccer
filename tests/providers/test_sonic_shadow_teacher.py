from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation, SonicNavigationConfig


def teacher(monkeypatch):
    nav = object.__new__(G1SonicNavigation)
    nav.config = SonicNavigationConfig(maximum_frames=300, model_variant="low_latency")
    nav._faulted = nav._retired = nav._batch_pending = False
    nav._next_frame = 20
    nav.backend = SimpleNamespace(
        action=np.zeros(29, dtype=np.float32),
        default_angles=np.zeros(29),
        _action_scale=np.ones(29),
        kp=np.ones(29),
        kd=np.ones(29),
    )
    calls = []

    def prepare(obs):
        calls.append(("observe", nav.backend.action.copy()))
        return None, 20

    def tick(state, frame):
        nav.backend.action = np.ones(29, dtype=np.float32)
        return np.ones(29)

    monkeypatch.setattr(nav, "_prepare_policy_observation", prepare)
    nav.backend.navigation_tick = tick
    return nav, calls


def test_advice_is_not_executed_history_and_actual_action_is_committed(monkeypatch):
    nav, calls = teacher(monkeypatch)
    advice = nav.prepare_shadow_action(None)
    assert nav._next_frame == 20 and nav._batch_pending and not advice.flags.writeable
    np.testing.assert_array_equal(advice, np.ones(29))
    actual = np.full(29, 0.25, dtype=np.float32)
    target = nav.commit_batched_action(actual)
    assert nav._next_frame == 21 and not nav._batch_pending
    np.testing.assert_array_equal(nav.backend.action, actual)
    np.testing.assert_array_equal(target.target_rad, actual)
    nav.prepare_shadow_action(None)
    np.testing.assert_array_equal(calls[-1][1], actual)


def test_missing_commit_latches_shadow_teacher_not_silent_history_reuse(monkeypatch):
    nav, _ = teacher(monkeypatch)
    nav.prepare_shadow_action(None)
    with pytest.raises(ValueError, match="latched off"):
        nav.prepare_shadow_action(None)
    assert nav._faulted


def test_bad_actual_action_latches_both_regular_batch_and_shadow_commit(monkeypatch):
    nav, _ = teacher(monkeypatch)
    nav.prepare_shadow_action(None)
    with pytest.raises(ValueError):
        nav.commit_batched_action(np.full(29, np.nan, dtype=np.float32))
    assert nav._faulted and nav._next_frame == 20
