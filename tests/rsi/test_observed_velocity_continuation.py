import numpy as np
import pytest
from test_continuous_velocity_stream import stream
from test_g1_velocity_policy import state

from rosclaw_soccer.providers.g1.velocity_policy import target29
from rosclaw_soccer.rsi.observed_velocity_continuation import (
    ObservedVelocityContinuation,
    exit_after_entry_abstention,
)


def test_velocity_identity_branch_preserves_original_action_and_phase(monkeypatch):
    original, owned = stream(monkeypatch), stream(monkeypatch)
    continuation = ObservedVelocityContinuation(owned)
    q, v = state()
    command = np.array([0.6, 0.1, 0.0], np.float32)
    for frame in range(80):
        expected = original.propose(q, v, np.zeros(3), command, frame=frame)
        actual = continuation.propose(q, v, np.zeros(3), command, frame=frame)
        receipt = continuation.observe_executed(actual, completed_control=frame + 1)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(owned.last_action, original.last_action)
        np.testing.assert_array_equal(owned.last_observation, original.last_observation)
        assert owned.phase == original.phase
        assert receipt["original_shadow_action_retained"]
        assert receipt["maximum_target_decode_error_rad"] == 0


def test_other_skill_actual_history_not_shadow_output_is_used_next_tick(monkeypatch):
    owned = stream(monkeypatch)
    continuation = ObservedVelocityContinuation(owned)
    q, v = state()
    command = np.array([0.6, 0.0, 0.0], np.float32)
    continuation.propose(q, v, np.zeros(3), command, frame=0)
    action = np.linspace(-0.5, 0.5, 29, dtype=np.float32)
    actual_target = target29(action, owned.params)
    before = actual_target.copy()
    receipt = continuation.observe_executed(actual_target, completed_control=1)
    np.testing.assert_array_equal(actual_target, before)
    assert not receipt["original_shadow_action_retained"]
    assert receipt["maximum_target_decode_error_rad"] <= 5e-7
    assert not receipt["phase_reset"] and not receipt["physical_target_modified"]
    history = owned.last_action.copy()
    phase = owned.phase
    continuation.propose(q, v, np.zeros(3), command, frame=1)
    np.testing.assert_array_equal(owned.last_observation[69:], history)
    assert owned.phase != phase


def test_missing_commit_and_bad_commit_latch_fault_without_reinitialization(monkeypatch):
    owned = stream(monkeypatch)
    continuation = ObservedVelocityContinuation(owned)
    q, v = state()
    proposal = continuation.propose(q, v, np.zeros(3), np.zeros(3, np.float32), frame=0)
    with pytest.raises(ValueError):
        continuation.propose(q, v, np.zeros(3), np.zeros(3, np.float32), frame=1)
    with pytest.raises(ValueError):
        continuation.observe_executed(proposal, completed_control=2)
    assert continuation.failed and owned.neural_calls == 1
    with pytest.raises(ValueError):
        continuation.observe_executed(proposal, completed_control=1)


def test_used_stream_cannot_be_reset_for_handoff(monkeypatch):
    owned = stream(monkeypatch)
    q, v = state()
    owned.propose(q, v, np.zeros(3), np.zeros(3, np.float32), frame=0)
    with pytest.raises(ValueError):
        ObservedVelocityContinuation(owned)


@pytest.mark.parametrize(
    "target",
    [
        np.zeros(28, np.float64),
        np.zeros(29, np.float32),
        np.full(29, np.nan),
        np.full(29, np.inf),
        np.full(29, 10.01),
    ],
)
def test_invalid_actual_observation_latches_without_clipping_or_retry(monkeypatch, target):
    owned = stream(monkeypatch)
    continuation = ObservedVelocityContinuation(owned)
    q, v = state()
    proposed = continuation.propose(q, v, np.zeros(3), np.zeros(3, np.float32), frame=0)
    phase, action = owned.phase, owned.last_action.copy()
    with pytest.raises(ValueError):
        continuation.observe_executed(target, completed_control=1)
    assert continuation.failed and owned.phase == phase
    np.testing.assert_array_equal(owned.last_action, action)
    with pytest.raises(ValueError):
        continuation.observe_executed(proposed, completed_control=1)


@pytest.mark.parametrize("passed", [False, True])
def test_exit_gate_uses_current_prediction_not_source_labels(passed):
    event = dict(selected_window=dict(window_pass=passed))
    assert exit_after_entry_abstention([event]) is (not passed)
    event.update(course_index=1, source_outcome="arbitrary ignored label")
    assert exit_after_entry_abstention([event]) is (not passed)


@pytest.mark.parametrize(
    "events",
    [
        [],
        [None],
        [{}, {}],
        [{}],
        [dict(selected_window={})],
        [dict(selected_window=dict(window_pass=1))],
    ],
)
def test_exit_gate_fails_closed_without_typed_current_prediction(events):
    with pytest.raises(ValueError):
        exit_after_entry_abstention(events)
