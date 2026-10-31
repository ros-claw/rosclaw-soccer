from dataclasses import replace

import numpy as np
import pytest
from test_s215_near_ball_residual import policy, samples

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.training.near_ball_residual_ppo import update_private_actors


def test_player_context_zero_lift_preserves_nonzero_parent_and_roundtrips(tmp_path):
    original = policy()
    rng = np.random.default_rng(85)
    weights = {k: v.copy() for k, v in original.weights.items()}
    weights["w2"] = rng.normal(0, 0.1, weights["w2"].shape)
    weights["wv"] = rng.normal(0, 0.1, weights["wv"].shape)
    original = replace(original, weights=weights)
    upgraded = original.with_player_context()
    obs = rng.normal(size=(8, 56))
    extended = np.c_[obs, rng.normal(size=(8, 5))]
    for before, after in zip(
        original.act(obs, np.random.default_rng(8), explore=True),
        upgraded.act(extended, np.random.default_rng(8), explore=True),
        strict=True,
    ):
        np.testing.assert_allclose(before, after, rtol=0, atol=1e-14)
    assert upgraded.parent_hash == original.policy_hash
    path = tmp_path / "player-context.npz"
    upgraded.save(path)
    assert NearBallResidualPolicy.load(path).policy_hash == upgraded.policy_hash
    with pytest.raises(ValueError):
        upgraded.with_player_context()


def test_player_context_features_can_train_only_selected_private_actor():
    pytest.importorskip("torch")
    original = policy()
    upgraded = original.with_player_context()
    trace = samples(original)
    rng = np.random.default_rng(39)
    obs = np.concatenate((trace["residual_observations"], rng.normal(size=(40, 8, 5))), axis=2)
    actions, logps, values = zip(*(upgraded.act(x, rng, explore=True) for x in obs), strict=True)
    trace.update(
        residual_observations=obs,
        residual_latent=np.asarray(actions),
        residual_log_probability=np.asarray(logps),
        residual_value=np.asarray(values),
        residual_observation_contract_code=np.full(40, 3, dtype=np.int64),
    )
    child, rows = update_private_actors(upgraded, [trace])
    assert child.observation_contract == "player_context_v3"
    assert rows[0]["updated"]
    assert np.any(child.weights["w1"][0, -5:] != 0)
    np.testing.assert_array_equal(child.weights["w1"][1:], upgraded.weights["w1"][1:])
    del trace["residual_observation_contract_code"]
    with pytest.raises(ValueError, match="contracts differ"):
        update_private_actors(upgraded, [trace])
