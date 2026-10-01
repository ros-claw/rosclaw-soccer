import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.stochastic_step_execution import (
    delta_at_frame,
    latent_sample,
    make_preview,
    make_sampling_view,
)
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_step_motor_execution import body
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_reproducible_latent_sampling_and_physical_shields(model):  # noqa: F811
    view = make_sampling_view(model, seed=316, std=0.1)
    sample, logp = latent_sample(view, np.zeros(134), 30)
    again, second_logp = latent_sample(view, np.zeros(134), 30)
    assert np.array_equal(sample, again) and logp == second_logp
    assert np.isfinite(logp)
    other, _ = latent_sample(view, np.zeros(134), 31)
    assert not np.array_equal(sample, other)
    policy = make_preview(view)
    observation = body()
    boundary = dict(
        frame=30,
        nominal_target=np.zeros(29),
        baseline=np.full(12, 1.2),
        limits=np.tile([-1.0, 1.0], (12, 1)),
        previous=np.zeros(12),
        previous_contact_forces=np.zeros(6),
    )
    first = delta_at_frame(policy, observation, **boundary)
    assert np.max(np.abs(first)) <= 0.012
    assert np.all(first <= 0)
    observation["joint_position_rad"][31:] = np.nan
    assert np.array_equal(first, delta_at_frame(policy, observation, **boundary))
    assert policy["stochastic_motor_proof"]["promotion_authorized"] is False


@pytest.mark.parametrize("seed,std", [(-1, 0.1), (True, 0.1), (1, np.nan), (1, 0.2)])
def test_invalid_exploration_contract_rejected(model, seed, std):  # noqa: F811
    with pytest.raises(ValueError):
        make_sampling_view(model, seed=seed, std=std)


def test_resealed_exploration_cannot_mutate_mean_policy(model):  # noqa: F811
    view = copy.deepcopy(make_sampling_view(model, seed=316, std=0.1))
    view["actor"]["layers"][0]["bias"][0] += 0.1
    view["model_hash"] = hash_json({k: v for k, v in view.items() if k != "model_hash"})
    with pytest.raises(ValueError, match="mean policy weights"):
        make_preview(view)
