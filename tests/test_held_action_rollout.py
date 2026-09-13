import numpy as np
import pytest

from rosclaw_soccer.training.held_action_rollout import coarsen_held_action_rollout


def rollout():
    data = {
        key: np.zeros((10, 2, 3 if key == "obs" else 2), np.float32)
        if key in ("obs", "raw")
        else np.zeros((10, 2), np.float32)
        for key in ("obs", "raw", "logp", "value", "reward", "alive", "next_alive")
    }
    data["alive"][:] = 1
    data["next_alive"][:-1] = 1
    data["raw"][:5] = 0.1
    data["raw"][5:] = 0.2
    data["obs"][:, :, 1] = 1
    data["reward"][:] = np.arange(20).reshape(10, 2)
    return data


def coarsen(data, **changes):
    options = dict(period=5, gamma=0.995, observation_size=3, action_size=2, learning_feature=1)
    options.update(changes)
    return coarsen_held_action_rollout(data, **options)


def test_exact_discounted_return_and_private_arrays():
    data = rollout()
    result = coarsen(data)
    expected = sum(0.995**i * data["reward"][i].astype(np.float64) for i in range(10))
    actual = result["reward"][0] + 0.995**5 * result["reward"][1]
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=2e-6)
    for key in result:
        assert not np.shares_memory(result[key], data[key])


def test_mid_decision_termination():
    data = rollout()
    data["next_alive"][7, 0] = 0
    for value in data.values():
        value[8:, 0] = 0
    result = coarsen(data)
    assert result["alive"][1, 0] == 1 and result["next_alive"][1, 0] == 0
    assert result["reward"][1, 0] == np.float32(
        sum(0.995**i * float(data["reward"][5 + i, 0]) for i in range(3))
    )


def test_period_one_is_identity():
    data = rollout()
    result = coarsen(data, period=1)
    for key in data:
        np.testing.assert_array_equal(result[key], data[key])


@pytest.mark.parametrize(
    "changes",
    [
        dict(period=True),
        dict(period=0),
        dict(period=11),
        dict(period=3),
        dict(gamma=float("nan")),
        dict(gamma=True),
        dict(gamma=0),
        dict(gamma=1.1),
        dict(observation_size=0),
        dict(action_size=65),
        dict(learning_feature=3),
    ],
)
def test_invalid_contract(changes):
    with pytest.raises(ValueError):
        coarsen(rollout(), **changes)


@pytest.mark.parametrize(
    "kind", ["action", "window", "nan", "revival", "terminal", "dead_reward", "overflow"]
)
def test_invalid_rollout(kind):
    data = rollout()
    if kind == "action":
        data["raw"][2, 0, 0] = 3
    elif kind == "window":
        data["obs"][2, 0, 1] = 0
    elif kind == "nan":
        data["reward"][2, 0] = np.nan
    elif kind == "revival":
        data["next_alive"][2, 0] = 0
    elif kind == "terminal":
        data["next_alive"][-1] = 1
    elif kind == "overflow":
        data["reward"][:] = np.finfo(np.float32).max
    else:
        data["next_alive"][7, 0] = 0
        for value in data.values():
            value[8:, 0] = 0
        data["reward"][8, 0] = 1
    with pytest.raises(ValueError):
        coarsen(data)
