import numpy as np
import pytest

from rosclaw_soccer.training.team_initial_selector import (
    InitialTeamSelector,
    fit_initial_team_selector,
    initial_team_features,
)

IDS = tuple(f"player.{i}" for i in range(8))


def test_causal_features_accept_only_one_synchronized_boundary():
    q, v = np.zeros((8, 43)), np.zeros((8, 41))
    ids = IDS
    q[:, 36:39] = (2.0, 1.0, 0.11)
    v[:, 35:37] = (0.2, -0.1)
    q[:, 0] = np.arange(8)
    first = initial_team_features(q, v, agent_ids=ids)
    assert first.shape == (34,)
    np.testing.assert_allclose(first[:2], (-2, -1))
    np.testing.assert_allclose(first[-2:], (0.2, -0.1))
    q[4, 36] += 1
    with pytest.raises(ValueError, match="synchronized"):
        initial_team_features(q, v, agent_ids=ids)
    with pytest.raises(ValueError):
        initial_team_features(np.zeros((2, 8, 43)), v, agent_ids=ids)
    with pytest.raises(ValueError):
        initial_team_features(np.zeros((8, 43)), v, agent_ids=tuple(reversed(ids)))


def test_training_is_deterministic_and_preserves_caller_arrays():
    x = np.zeros((8, 34))
    x[:, 0] = np.arange(8) - 3.5
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    source = x.copy()
    model = fit_initial_team_selector(x, y, seed=14, agent_ids=IDS)
    repeat = fit_initial_team_selector(x, y, seed=14, agent_ids=IDS)
    np.testing.assert_array_equal(x, source)
    np.testing.assert_array_equal(model.weight, repeat.weight)
    np.testing.assert_array_equal(model.propose(x, agent_ids=IDS), y)
    with pytest.raises(ValueError):
        model.propose(x, agent_ids=tuple(reversed(IDS)))
    with pytest.raises(ValueError):
        model.weight[0] = 1


@pytest.mark.parametrize(
    "labels",
    [
        np.array([0, 0, 0, 0]),
        np.array([0, 1, 1, 1]),
        np.array([0, 0, 1, 2]),
        np.array([0.0, 0.0, 1.0, 1.0]),
    ],
)
def test_no_ambiguous_or_single_class_labels(labels):
    with pytest.raises(ValueError):
        fit_initial_team_selector(np.zeros((4, 34)), labels, seed=1, agent_ids=IDS)


def test_bad_parameters_and_inputs_fail_closed():
    with pytest.raises(ValueError):
        InitialTeamSelector(np.zeros(34), np.zeros(34), np.zeros(34), 0.0, IDS)
    model = InitialTeamSelector(np.zeros(34), np.ones(34), np.zeros(34), 0.0, IDS)
    with pytest.raises(ValueError):
        model.propose(np.full((1, 34), np.nan), agent_ids=IDS)


def test_pickle_free_roundtrip_and_no_overwrite(tmp_path):
    path = tmp_path / "weights.npz"
    model = InitialTeamSelector(np.zeros(34), np.ones(34), np.arange(34.0), 0.3, IDS)
    model.save(path)
    restored = InitialTeamSelector.load(path)
    np.testing.assert_array_equal(restored.weight, model.weight)
    assert restored.agent_ids == IDS
    with pytest.raises(FileExistsError):
        model.save(path)
    wrong = tmp_path / "wrong.npz"
    np.savez(wrong, arbitrary=np.zeros(1))
    with pytest.raises(ValueError):
        InitialTeamSelector.load(wrong)
