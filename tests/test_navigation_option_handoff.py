from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_s338_sonic_history_handoff import (
    assert_history_equal,
    fresh_navigation,
    history_copy,
    navigation_fixture,
)


def options():
    source, context = navigation_fixture()
    destination, _, calls = fresh_navigation()
    destination.contract_hash = "sha256:" + "c" * 64
    observation = replace(context["source_observation"], navigation_command=(0.3, 0.0, 0.0))
    return source, destination, observation, calls


def test_atomic_option_handoff_preserves_history_action_and_global_clock():
    source, destination, observation, calls = options()
    before = history_copy(source.backend)
    last_action = source.backend.action.copy()
    receipt = source.handoff_to_navigation(destination, observation=observation)
    assert receipt.frame == 120 and receipt.agent_id == source.agent_id
    assert source._retired and source._next_frame == 121
    assert destination._ready_from_handoff and not destination._ready_from_observation
    assert destination._origin_frame == destination._next_frame == 120
    assert calls["reset"] == 1 and calls["frames"] == []
    assert_history_equal(before[1:], list(destination.backend._history)[:-1])
    np.testing.assert_array_equal(destination.backend.action, last_action)
    imported = history_copy(destination.backend)
    destination.propose(observation)
    assert calls["reset"] == 1 and calls["frames"] == [0]
    assert_history_equal(imported, destination.backend._history)
    destination.propose(replace(observation, frame=121, time_sec=2.42))
    assert calls["frames"] == [0, 1]
    with pytest.raises(ValueError):
        source.propose(observation)


@pytest.mark.parametrize(
    "fault", ["source_pending", "destination_pending", "identity", "used", "foundation", "frame"]
)
def test_failed_transition_latches_both_and_never_decodes(fault):
    source, destination, observation, calls = options()
    if fault == "source_pending":
        source._batch_pending = True
    elif fault == "destination_pending":
        destination._batch_pending = True
    elif fault == "identity":
        destination.agent_id = "blue.finisher"
    elif fault == "used":
        destination._next_frame = 1
    elif fault == "foundation":
        destination.backend.qualification = SimpleNamespace(qualification_hash="sha256:" + "b" * 64)
    elif fault == "frame":
        observation = replace(observation, frame=121, time_sec=2.42)
    with pytest.raises(ValueError, match="both options latched"):
        source.handoff_to_navigation(destination, observation=observation)
    assert source._faulted and destination._faulted
    assert calls["frames"] == []
    with pytest.raises(ValueError):
        destination.propose(observation)


def test_destination_first_tick_remains_bound_to_the_exact_observation():
    source, destination, observation, calls = options()
    source.handoff_to_navigation(destination, observation=observation)
    with pytest.raises(ValueError):
        destination.propose(replace(observation, navigation_command=(0.4, 0.0, 0.0)))
    assert destination._faulted and calls["frames"] == []
