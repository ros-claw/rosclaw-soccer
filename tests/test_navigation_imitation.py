import numpy as np
import pytest

from rosclaw_soccer.training.local_navigation import build_local_navigation_actor_critic
from rosclaw_soccer.training.navigation_imitation import fit_navigation_demonstration


def examples():
    torch = pytest.importorskip("torch")
    torch.manual_seed(2122)
    model = build_local_navigation_actor_critic()
    weights = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
    obs = np.zeros((8, 39), np.float32)
    target = np.tile(np.array((0.1, 0.05, 0), np.float32), (8, 1))
    return weights, obs, target


def test_imitation_is_exact_replay_and_does_not_mutate_parent_or_critic():
    parent, obs, target = examples()
    original = {k: v.copy() for k, v in parent.items()}
    child, stats = fit_navigation_demonstration(parent, obs, target, steps=32)
    replay, same = fit_navigation_demonstration(parent, obs, target, steps=32)
    assert stats == same and stats["mse_after"] < stats["mse_before"]
    assert all(np.array_equal(child[k], replay[k]) for k in child)
    assert all(np.array_equal(parent[k], original[k]) for k in parent)
    assert all(np.array_equal(child[k], parent[k]) for k in child if not k.startswith("actor."))


@pytest.mark.parametrize("fault", ["nan", "authority", "unaligned", "parent", "steps"])
def test_invalid_training_inputs_rejected(fault):
    parent, obs, target = examples()
    kw = {}
    if fault == "nan":
        obs[0, 0] = np.nan
    elif fault == "authority":
        target[0, :2] = 0.25
    elif fault == "unaligned":
        target = target[:-1]
    elif fault == "parent":
        parent["logstd"][0] = np.inf
    else:
        kw["steps"] = True
    with pytest.raises(ValueError):
        fit_navigation_demonstration(parent, obs, target, **kw)
