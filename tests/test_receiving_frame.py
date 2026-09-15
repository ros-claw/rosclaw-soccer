from dataclasses import replace

import numpy as np
import pytest
from test_s368_recurrent_receiver import CONFIG, POLICY, observation

from rosclaw_soccer.providers.g1.recurrent_receiver import G1RecurrentReceiver
from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic
from rosclaw_soccer.training.receiving_frame import (
    CANONICAL_RECEIVER_CONTRACT,
    canonical_receiving_features,
)


def test_half_turn_exact_field_mapping_involution_and_private_copy():
    x = np.arange(270, dtype=np.float32).reshape(2, 135)
    before = x.copy()
    y = canonical_receiving_features(x, half_turn=True)
    expected = x.copy()
    for start in (61, 67, 70, 133):
        expected[:, start : start + 2] *= -1
    np.testing.assert_array_equal(y, expected)
    np.testing.assert_array_equal(canonical_receiving_features(y, half_turn=True), x)
    np.testing.assert_array_equal(x, before)
    z = canonical_receiving_features(x[0], half_turn=False)
    np.testing.assert_array_equal(z, x[0])
    assert not np.shares_memory(z, x)


@pytest.mark.parametrize(
    "x",
    [
        np.zeros(135, dtype=np.float64),
        np.zeros(134, dtype=np.float32),
        np.zeros((1, 1, 135), dtype=np.float32),
        np.zeros((0, 135), dtype=np.float32),
        np.full(135, np.nan, dtype=np.float32),
        np.full(135, np.inf, dtype=np.float32),
        np.zeros((4097, 135), dtype=np.float32),
    ],
)
def test_bad_features_rejected(x):
    with pytest.raises(ValueError):
        canonical_receiving_features(x, half_turn=False)


@pytest.mark.parametrize("flag", [0, 1, None, "false", np.bool_(True)])
def test_explicit_boolean_required(flag):
    with pytest.raises(ValueError):
        canonical_receiving_features(np.zeros(135, dtype=np.float32), half_turn=flag)


def build(tmp_path, contract, flag=None):
    import torch

    torch.manual_seed(42)
    actor = build_ball_residual_actor_critic(observation_size=135)
    with torch.no_grad():
        actor.actor[-1].weight.fill_(0.01)
    path = tmp_path / "heading.npz"
    np.savez_compressed(path, **{k: v.detach().numpy() for k, v in actor.state_dict().items()})
    return G1RecurrentReceiver(
        path,
        agent_id="blue.playmaker",
        expected_actor_hash=hash_bytes(path.read_bytes()),
        foundation_hash=POLICY,
        foundation_config_hash=CONFIG,
        observation_contract=contract,
        canonical_half_turn=flag,
    )


def test_false_preserves_legacy_inputs_actions_but_has_distinct_bound_contract(tmp_path):
    old = build(tmp_path, "recurrent_receiver_world_heading_135_float32.v2")
    new = build(tmp_path, CANONICAL_RECEIVER_CONTRACT, False)
    old.begin_skill(observation())
    new.begin_skill(observation())
    for i in range(4):
        obs = observation(415 + i, 8.3 + 0.02 * i)
        assert old.propose(obs) == new.propose(obs)
        np.testing.assert_array_equal(old.last_observation, new.last_observation)
    assert old.contract_hash != new.contract_hash
    assert new.contract_hash != build(tmp_path, CANONICAL_RECEIVER_CONTRACT, True).contract_hash


def test_physical_half_turn_preserves_joint_and_body_fields(tmp_path):
    base = build(tmp_path, CANONICAL_RECEIVER_CONTRACT, False)
    turned = build(tmp_path, CANONICAL_RECEIVER_CONTRACT, True)
    obs = observation()
    vel = list(obs.qvel)
    vel[:6] = [0.3, -0.2, 0.1, 0.2, 0.1, -0.3]
    vel[35:38] = [1.0, 2.0, 0.4]
    obs = replace(obs, qvel=tuple(vel))
    q = list(obs.qpos)
    q[0], q[1], q[36], q[37] = -q[0], -q[1], -q[36], -q[37]
    q[3], q[6] = 0.0, 1.0
    for i in (0, 1, 35, 36):
        vel[i] *= -1
    rotated = replace(obs, qpos=tuple(q), qvel=tuple(vel))
    base.begin_skill(obs)
    turned.begin_skill(rotated)
    a, b = base.propose(obs), turned.propose(rotated)
    np.testing.assert_allclose(base.last_observation, turned.last_observation, atol=1e-6)
    np.testing.assert_allclose(a.target_rad, b.target_rad, atol=1e-7)


@pytest.mark.parametrize(
    "contract,flag",
    [
        (CANONICAL_RECEIVER_CONTRACT, None),
        (CANONICAL_RECEIVER_CONTRACT, 1),
        ("recurrent_receiver_world_heading_135_float32.v2", False),
        ("recurrent_receiver_followup_target_138_float32.v3", True),
    ],
)
def test_no_silent_contract_reinterpretation(tmp_path, contract, flag):
    with pytest.raises(ValueError):
        build(tmp_path, contract, flag)


def test_canonical_training_samples_bind_transformed_inputs_and_likelihoods(tmp_path):
    import torch

    from rosclaw_soccer.training.receiving_sampler import ReceivingSampler

    reference = build(tmp_path, CANONICAL_RECEIVER_CONTRACT, True)
    path = tmp_path / "heading.npz"
    sampled = ReceivingSampler(
        path,
        seed=2228,
        agent_id="blue.playmaker",
        expected_actor_hash=hash_bytes(path.read_bytes()),
        foundation_hash=POLICY,
        foundation_config_hash=CONFIG,
        observation_contract=CANONICAL_RECEIVER_CONTRACT,
        canonical_half_turn=True,
    )
    obs = observation()
    for m in (reference, sampled):
        m.begin_skill(obs)
        m.propose(obs)
    rows = sampled.sampled_rollout()
    np.testing.assert_array_equal(rows["obs"], reference.last_observation)
    with torch.no_grad():
        mean, value = sampled._actor(torch.from_numpy(rows["obs"]))
        distribution = torch.distributions.Normal(
            mean, sampled._actor.logstd.clamp(-2.5, -0.3).exp()
        )
        expected = distribution.log_prob(torch.from_numpy(rows["raw"])).sum(1)
    np.testing.assert_array_equal(expected.numpy(), rows["logp"])
    np.testing.assert_array_equal(value.numpy(), rows["value"])
    assert sampled.contract_hash != reference.contract_hash
