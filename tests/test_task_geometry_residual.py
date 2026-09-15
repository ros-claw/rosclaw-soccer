from dataclasses import replace

import numpy as np
import pytest
from test_s215_near_ball_residual import policy, samples

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.training.near_ball_residual_ppo import update_private_actors


def test_geometry_migration_preserves_nonzero_actor_and_value_and_roundtrips(tmp_path):
    original = policy()
    weights = {k: v.copy() for k, v in original.weights.items()}
    rng = np.random.default_rng(35)
    weights["w2"] = rng.normal(0, 0.1, weights["w2"].shape)
    weights["wv"] = rng.normal(0, 0.1, weights["wv"].shape)
    original = replace(original, weights=weights)
    upgraded = original.with_task_geometry()
    obs = rng.normal(0, 0.1, (8, 56))
    extended = np.c_[obs, rng.normal(size=(8, 2))]
    for before, after in zip(
        original.act(obs, np.random.default_rng(7), explore=True),
        upgraded.act(extended, np.random.default_rng(7), explore=True),
        strict=True,
    ):
        np.testing.assert_allclose(before, after, rtol=0, atol=1e-14)
    assert upgraded.parent_hash == original.policy_hash
    assert upgraded.policy_hash != original.policy_hash
    destination = tmp_path / "geometry.npz"
    upgraded.save(destination)
    assert NearBallResidualPolicy.load(destination).policy_hash == upgraded.policy_hash
    with pytest.raises(ValueError):
        upgraded.act(obs, rng, explore=False)
    with pytest.raises(ValueError):
        upgraded.with_task_geometry()


def test_geometry_features_are_trainable_and_updates_retain_contract():
    pytest.importorskip("torch")
    original = policy()
    upgraded = original.with_task_geometry()
    trace = samples(original)
    rng = np.random.default_rng(18)
    obs = np.concatenate((trace["residual_observations"], rng.normal(size=(40, 8, 2))), axis=2)
    actions, logps, values = zip(*(upgraded.act(x, rng, explore=True) for x in obs), strict=True)
    trace.update(
        residual_observations=obs,
        residual_latent=np.asarray(actions),
        residual_log_probability=np.asarray(logps),
        residual_value=np.asarray(values),
        residual_observation_contract_code=np.full(40, 2, dtype=np.int64),
    )
    child, rows = update_private_actors(upgraded, [trace])
    assert child.observation_contract == "task_geometry_v2"
    assert rows[0]["updated"]
    assert np.any(child.weights["w1"][0, -2:] != 0)
    np.testing.assert_array_equal(child.weights["w1"][1:], upgraded.weights["w1"][1:])
    del trace["residual_observation_contract_code"]
    with pytest.raises(ValueError, match="contracts differ"):
        update_private_actors(upgraded, [trace])
