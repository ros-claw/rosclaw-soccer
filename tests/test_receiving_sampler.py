import numpy as np
import pytest
from test_s368_recurrent_receiver import CONFIG, POLICY, motor, observation

from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.training.receiving_sampler import ReceivingSampler


def sampler(tmp_path, seed):
    motor(tmp_path)
    path = tmp_path / "receiver.npz"
    return ReceivingSampler(
        path,
        seed=seed,
        agent_id="blue.playmaker",
        expected_actor_hash=hash_bytes(path.read_bytes()),
        foundation_hash=POLICY,
        foundation_config_hash=CONFIG,
    )


def test_actual_seeded_samples_and_likelihoods_are_replayable(tmp_path):
    import torch

    first = sampler(tmp_path, 4)
    # Reuse the same content-bound weights, not another randomly initialized critic.
    second = ReceivingSampler(
        tmp_path / "receiver.npz",
        seed=4,
        agent_id="blue.playmaker",
        expected_actor_hash=hash_bytes((tmp_path / "receiver.npz").read_bytes()),
        foundation_hash=POLICY,
        foundation_config_hash=CONFIG,
    )
    for m in (first, second):
        m.begin_skill(observation())
    for frame in range(5):
        o = observation(415 + frame, 8.3 + 0.02 * frame)
        a, b = first.propose(o), second.propose(o)
        assert a == b
        assert max(abs(x - 0.25) for x in a.target_rad) <= 0.25
    rows = first.sampled_rollout()
    for key, values in rows.items():
        np.testing.assert_array_equal(values, second.sampled_rollout()[key])
    with torch.no_grad():
        mean, value = first._actor(torch.tensor(rows["obs"]))
        logp = (
            torch.distributions.Normal(mean, first._actor.logstd.clamp(-2.5, -0.3).exp())
            .log_prob(torch.tensor(rows["raw"]))
            .sum(1)
        )
    np.testing.assert_allclose(logp.numpy(), rows["logp"], atol=1e-5)
    np.testing.assert_allclose(value.numpy(), rows["value"], atol=1e-6)
    assert np.any(rows["raw"] != 0)
    rows["raw"][:] = 0
    assert np.any(first.sampled_rollout()["raw"] != 0)


@pytest.mark.parametrize("seed", [True, -1, 2**32, 0.5])
def test_bad_seed_fails_before_loading_model(tmp_path, seed):
    with pytest.raises(ValueError, match="seed"):
        ReceivingSampler(
            tmp_path / "missing",
            seed=seed,
            agent_id="blue.playmaker",
            expected_actor_hash=POLICY,
            foundation_hash=POLICY,
            foundation_config_hash=CONFIG,
        )
