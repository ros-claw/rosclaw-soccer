from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.amp_history import AmpFrameSpec, AmpObservationHistory


def spec():
    return AmpFrameSpec(
        tuple(reversed(range(29))), (0.0,) * 29, (-1.5, -0.01, -1.57), (3.0, 0.01, 1.57)
    )


def body():
    return dict(
        joint_position=np.arange(29, dtype=float) / 100,
        joint_velocity=np.arange(29, dtype=float),
        projected_gravity=np.array([0.0, 0.0, -1.0]),
        angular_velocity=np.zeros(3),
    )


def started():
    h = AmpObservationHistory(spec())
    h.begin(**body())
    return h


def test_mapping_history_and_action_order():
    h = started()
    obs = h.prepare(tick=0, command=np.array([1.0, 0.0, 0.0]), **body()).reshape(4, 96)
    np.testing.assert_array_equal(obs[:3, 6:9], np.zeros((3, 3)))
    assert obs[-1, 6] == np.float32(0.4)
    np.testing.assert_array_equal(obs[-1, 9:38], (np.arange(29)[::-1] / 100).astype(np.float32))
    raw = np.arange(29, dtype=np.float32)
    np.testing.assert_array_equal(h.commit_action(raw), raw[::-1] * 0.25)
    obs2 = h.prepare(tick=1, command=np.zeros(3), **body()).reshape(4, 96)
    np.testing.assert_array_equal(obs2[-1, 67:], raw)
    np.testing.assert_array_equal(obs2[:3], obs[1:])


def test_snapshot_round_trip_and_no_alias():
    h = started()
    for tick in range(10):
        h.prepare(tick=tick, command=np.array([3.0, 0.01, 1.57], dtype=np.float32), **body())
        h.commit_action(np.ones(29) * tick)
    snapshot = h.snapshot()
    with pytest.raises(ValueError):
        snapshot.history.setflags(write=True)
    restored = AmpObservationHistory(spec())
    restored.restore(snapshot)
    for tick in range(10, 20):
        args = dict(tick=tick, command=np.zeros(3), **body())
        np.testing.assert_array_equal(h.prepare(**args), restored.prepare(**args))
        np.testing.assert_array_equal(
            h.commit_action(np.ones(29)), restored.commit_action(np.ones(29))
        )
    assert snapshot.next_tick == 10
    with pytest.raises(AttributeError):
        h.spec = spec()


@pytest.mark.parametrize(
    "failure", ["skip", "double_prepare", "action", "reset", "nan", "gravity", "range"]
)
def test_bad_call_latches_invalid(failure):
    h = started()
    args = dict(tick=0, command=np.zeros(3), **body())
    if failure == "skip":
        args["tick"] = 1
    if failure == "double_prepare":
        h.prepare(**args)
    if failure == "nan":
        args["joint_velocity"][0] = np.nan
    if failure == "gravity":
        args["projected_gravity"][:] = 0
    if failure == "range":
        args["command"][0] = 4
    with pytest.raises(ValueError):
        if failure == "action":
            h.commit_action(np.zeros(29))
        elif failure == "reset":
            h.begin(**body())
        else:
            h.prepare(**args)
    with pytest.raises(RuntimeError):
        h.snapshot()


@pytest.mark.parametrize("corruption", ["hash", "gravity", "action", "command", "tick"])
def test_restore_rejects_corruption(corruption):
    snapshot = started().snapshot()
    if corruption == "hash":
        snapshot = replace(snapshot, spec_hash="wrong")
    elif corruption == "tick":
        snapshot = replace(snapshot, next_tick=True)
    else:
        history = snapshot.history.copy()
        history[0, {"gravity": 3, "action": 67, "command": 6}[corruption]] = 101
        snapshot = replace(snapshot, history=history)
    h = AmpObservationHistory(spec())
    with pytest.raises(ValueError):
        h.restore(snapshot)
    with pytest.raises(RuntimeError):
        h.snapshot()


def test_pending_snapshot_does_not_destroy_valid_action():
    h = started()
    h.prepare(tick=0, command=np.zeros(3), **body())
    with pytest.raises(ValueError):
        h.snapshot()
    h.commit_action(np.zeros(29))
    assert h.snapshot().next_tick == 1


def test_players_do_not_share_observation_or_action_memory():
    first, second = started(), started()
    before = second.snapshot()
    obs = first.prepare(tick=0, command=np.array([1.0, 0.0, 0.0]), **body())
    obs[:] = 0
    first.commit_action(np.ones(29))
    np.testing.assert_array_equal(second.snapshot().history, before.history)
    assert second.snapshot().next_tick == 0
    assert first.snapshot().history[-1, 6] == np.float32(0.4)


@pytest.mark.parametrize("raw", [np.zeros(28), np.full(29, np.nan), np.full(29, 101.0)])
def test_invalid_model_output_latches_failure(raw):
    h = started()
    h.prepare(tick=0, command=np.zeros(3), **body())
    with pytest.raises(ValueError):
        h.commit_action(raw)
    with pytest.raises(RuntimeError):
        h.snapshot()


@pytest.mark.parametrize(
    "change",
    [
        dict(joint_indices=(0,) * 29),
        dict(command_smoothing=1.0),
        dict(action_scale=float("nan")),
        dict(command_low=(1.0, 0.0, 0.0)),
    ],
)
def test_bad_config(change):
    with pytest.raises(ValueError):
        replace(spec(), **change)
