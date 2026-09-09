import collections
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_history_handoff import handoff_sonic_history
from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation


def fixture():
    q = np.zeros(43)
    q[2] = 0.75
    q[3] = q[39] = 1
    q[7:36] = G1SonicRunupController.default_angles
    q[36:39] = (0.6, -0.04, 0.115)
    v = np.zeros(41)
    v[0] = 1.3
    o = TeamMotorObservation(
        "red.finisher",
        120,
        2.4,
        "shoot",
        True,
        tuple(q.tolist()),
        tuple(v.tolist()),
        (7.5, 0.0, 1.0),
    )
    controllers = []
    for is_source in (True, False):
        c = object.__new__(G1SonicRunupController)
        c.qualification = SimpleNamespace(qualification_hash="sha256:" + "a" * 64)
        c._kp = np.ones(29)
        c._kd = np.ones(29)
        c._action_scale = np.ones(29)
        c.action = np.full(29, 0.1 if is_source else 0, dtype=np.float32)
        c.reference = np.zeros((60, 36))
        c.reference[:, 3] = 1
        entry = c._history_entry(SimpleNamespace(qpos=q, qvel=v), c.action)
        c._history = collections.deque(
            [tuple(a.copy() for a in entry) for _ in range(10)], maxlen=10
        )
        if is_source:
            c._history[0][0][0] += 0.1
        controllers.append(c)
    return dict(
        source=controllers[0],
        destination=controllers[1],
        source_observation=o,
        destination_observation=o,
    )


def history_copy(controller):
    return [[v.copy() for v in e] for e in controller._history]


def assert_history_equal(a, b):
    for aa, bb in zip(a, b, strict=True):
        for x, y in zip(aa, bb, strict=True):
            np.testing.assert_array_equal(x, y)


def test_exact_history_transfer_replay_no_reference_write_or_alias():
    c = fixture()
    expected = history_copy(c["source"])
    reference = c["destination"].reference.copy()
    receipt = handoff_sonic_history(**c)
    assert receipt == handoff_sonic_history(**fixture())
    assert receipt.agent_id == "red.finisher" and receipt.activation_ceiling == "SIM_ONLY"
    assert_history_equal(c["destination"]._history, expected)
    np.testing.assert_array_equal(c["destination"].reference, reference)
    c["source"]._history[0][0][0] += 3
    c["source"].action[0] += 1
    assert_history_equal(c["destination"]._history, expected)
    assert c["destination"].action[0] < 0.2


def test_inference_copy_translation_does_not_change_proprioception():
    c = fixture()
    q = np.array(c["destination_observation"].qpos)
    q[:2] += [3.0, -1.4]
    q[36:38] += [3.0, -1.4]
    c["destination_observation"] = replace(c["destination_observation"], qpos=tuple(q.tolist()))
    result = handoff_sonic_history(**c)
    assert result.history_hash == handoff_sonic_history(**fixture()).history_hash
    assert result.binding_hash != handoff_sonic_history(**fixture()).binding_hash


@pytest.mark.parametrize(
    "change",
    [
        "agent",
        "frame",
        "time",
        "foundation",
        "gain",
        "state",
        "source_stale",
        "destination_used",
        "missing_history",
        "nan_history",
        "bad_reference",
        "same_object",
    ],
)
def test_invalid_handoff_is_atomic(change):
    c = fixture()
    if change == "agent":
        c["destination_observation"] = replace(
            c["destination_observation"], agent_id="blue.finisher"
        )
    elif change == "frame":
        c["destination_observation"] = replace(c["destination_observation"], frame=121)
    elif change == "time":
        c["destination_observation"] = replace(c["destination_observation"], time_sec=2.42)
    elif change == "foundation":
        c["destination"].qualification = SimpleNamespace(qualification_hash="sha256:" + "b" * 64)
    elif change == "gain":
        c["destination"]._kp[0] += 0.1
    elif change == "state":
        v = list(c["destination_observation"].qvel)
        v[0] += 0.1
        c["destination_observation"] = replace(c["destination_observation"], qvel=tuple(v))
    elif change == "source_stale":
        c["source"]._history[-1][2][0] += 0.1
    elif change == "destination_used":
        c["destination"]._history[0][0][0] += 0.1
    elif change == "missing_history":
        c["source"]._history.pop()
    elif change == "nan_history":
        c["source"]._history[0][0][0] = np.nan
    elif change == "bad_reference":
        c["source"].reference[0, 0] = np.nan
    elif change == "same_object":
        c["source"] = c["destination"]
    before = history_copy(c["destination"])
    action = c["destination"].action.copy()
    with pytest.raises(ValueError):
        handoff_sonic_history(**c)
    assert_history_equal(before, c["destination"]._history)
    np.testing.assert_array_equal(action, c["destination"].action)


def test_duplicate_handoff_does_not_overwrite_an_active_destination():
    c = fixture()
    handoff_sonic_history(**c)
    before = history_copy(c["destination"])
    with pytest.raises(ValueError):
        handoff_sonic_history(**c)
    assert_history_equal(before, c["destination"]._history)


def test_duplicate_zero_action_history_is_also_rejected():
    c = fixture()
    c["source"].action[:] = 0
    c["source"]._history = collections.deque(
        [tuple(v.copy() for v in entry) for entry in c["destination"]._history], maxlen=10
    )
    handoff_sonic_history(**c)
    with pytest.raises(ValueError):
        handoff_sonic_history(**c)
