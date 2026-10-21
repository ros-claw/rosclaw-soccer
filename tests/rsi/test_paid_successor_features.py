import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.paid_successor_features import (
    successor_features,
    successor_student_config,
)


class Base:
    mean = torch.zeros(160)
    denominator = torch.ones(160)

    def forward_with_features(self, obs, state):
        after = (state[0] + 0.1, state[1] + 0.2)
        return obs[..., :29], after, after[0][-1].reshape(1, 1, 128)


def fixture(*, teacher=False):
    base = Base()
    obs = np.zeros((3, 160), np.float32)
    obs[:, 67] = [0.1, 0.2, 0.3]
    before, after = [], []
    state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
    for item in obs:
        before.append(tuple(t.numpy().copy() for t in state))
        _, state, _ = base.forward_with_features(torch.from_numpy(item).reshape(1, 1, 160), state)
        after.append(tuple(t.numpy().copy() for t in state))
    raw = dict(
        NN_obs=obs,
        NN_nominal_action=np.zeros((3, 29), np.float32),
        NN_action=np.ones((3, 29), np.float32) * (0.02 if teacher else 0),
        NN_h_before=np.stack([s[0] for s in before]),
        NN_c_before=np.stack([s[1] for s in before]),
        NN_h_after=np.stack([s[0] for s in after]),
        NN_c_after=np.stack([s[1] for s in after]),
    )
    return base, raw


def test_features_only_own_observations_and_recomputed_hidden_not_case_ids():
    base, raw = fixture(teacher=True)
    x, y = successor_features(base, raw, torch.ones(29) * 0.5, teacher=True)
    assert x.shape == (3, 288) and np.array_equal(x[:, :160], raw["NN_obs"])
    assert np.allclose(x[:, 160], [0.1, 0.2, 0.3]) and np.allclose(y, 0.01)


@pytest.mark.parametrize("fault", ["nominal", "memory", "nonzero_retention", "bound"])
def test_rejects_unverified_features_or_false_zero_labels(fault):
    base, raw = fixture()
    if fault == "nominal":
        raw["NN_nominal_action"][1, 0] = 0.1
    elif fault == "memory":
        raw["NN_h_before"][1, 0, 0, 0] = 10
    else:
        raw["NN_action"][1, 0] = 0.21 if fault == "bound" else 0.01
    with pytest.raises(ValueError):
        successor_features(base, raw, torch.ones(29), teacher=fault == "bound")


@pytest.mark.parametrize("seed", [True, -1, 3, "0"])
def test_only_predeclared_seeds_with_no_development_fitting(seed):
    with pytest.raises(ValueError):
        successor_student_config(seed)


def test_independent_initializations_share_same_frozen_protocol():
    configs = [successor_student_config(i) for i in range(3)]
    assert len({c["seed"] for c in configs}) == 3
    assert all(c["DEV_episodes_fit"] == 0 and c["frozen_base"] for c in configs)
