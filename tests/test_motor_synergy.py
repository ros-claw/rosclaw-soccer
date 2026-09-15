import numpy as np
import pytest
from test_s368_recurrent_receiver import CONFIG, POLICY, observation

from rosclaw_soccer.providers.g1.recurrent_receiver import G1RecurrentReceiver
from rosclaw_soccer.providers.g1.synergy_receiver import G1SynergyReceiver
from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic
from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig
from rosclaw_soccer.training.motor_synergy import MotorSynergyBasis


def basis():
    return MotorSynergyBasis(tuple(tuple(float(x) for x in row) for row in np.eye(29, 3)), POLICY)


def test_basis_is_bounded_and_zero_preserves_parent():
    b = basis()
    np.testing.assert_array_equal(b.decode(np.zeros(3, np.float32)), np.zeros(29, np.float32))
    assert np.linalg.norm(b.decode(np.full(3, 100.0, np.float32))) <= 0.2 * np.sqrt(3) + 1e-7
    with pytest.raises(ValueError):
        b.decode(np.array([1.0, np.nan, 0.0]))
    with pytest.raises(ValueError):
        MotorSynergyBasis(((1.0, 0.0, 0.0),) * 29, POLICY)
    with pytest.raises(ValueError):
        MotorSynergyBasis(b.matrix, "unbound")
    assert FullBodyPPOUpdateConfig(observation_size=138, action_size=3).action_size == 3


def models(tmp_path):
    torch = pytest.importorskip("torch")
    parent = build_ball_residual_actor_critic(observation_size=138)
    with torch.no_grad():
        parent.actor[-1].bias.fill_(0.3)
    latent = build_ball_residual_actor_critic(observation_size=138, action_size=3)
    for name, model in [("parent", parent), ("latent", latent)]:
        np.savez_compressed(
            tmp_path / f"{name}.npz",
            **{k: v.detach().numpy() for k, v in model.state_dict().items()},
        )
    return dict(
        agent_id="blue.playmaker",
        expected_actor_hash=hash_bytes((tmp_path / "parent.npz").read_bytes()),
        foundation_hash=POLICY,
        foundation_config_hash=CONFIG,
        observation_contract="recurrent_receiver_followup_target_138_float32.v3",
        followup_target_position_m=(-1.5, -0.8, 1.5),
    )


def receiver(tmp_path, kwargs, explore=False):
    return G1SynergyReceiver(
        tmp_path / "parent.npz",
        latent_weights=tmp_path / "latent.npz",
        expected_latent_hash=hash_bytes((tmp_path / "latent.npz").read_bytes()),
        synergy=basis(),
        seed=5,
        explore=explore,
        **kwargs,
    )


def test_zero_latent_matches_nonzero_parent_over_full_skill(tmp_path):
    kwargs = models(tmp_path)
    parent = G1RecurrentReceiver(tmp_path / "parent.npz", **kwargs)
    latent = receiver(tmp_path, kwargs)
    parent.begin_skill(observation())
    latent.begin_skill(observation())
    for i in range(100):
        o = observation(415 + i, 8.3 + i * 0.02)
        assert latent.propose(o) == parent.propose(o)
    rollout = latent.sampled_rollout()
    assert rollout["obs"].shape == (100, 138) and rollout["raw"].shape == (100, 3)
    assert not rollout["raw"].any()


def test_seeded_latent_replay_and_basis_tamper_fault_latches(tmp_path):
    kwargs = models(tmp_path)
    a, b = receiver(tmp_path, kwargs, True), receiver(tmp_path, kwargs, True)
    for model in (a, b):
        model.begin_skill(observation())
    assert a.propose(observation()) == b.propose(observation())
    for k in a.sampled_rollout():
        np.testing.assert_array_equal(a.sampled_rollout()[k], b.sampled_rollout()[k])
    object.__setattr__(a._synergy, "raw_amplitude", 0.3)
    with pytest.raises(ValueError, match="basis changed"):
        a.propose(observation(416, 8.32))
    with pytest.raises(ValueError, match="unfaulted"):
        a.propose(observation(417, 8.34))


def test_invalid_action_dimension_is_not_a_joint_model():
    pytest.importorskip("torch")
    with pytest.raises(ValueError):
        build_ball_residual_actor_critic(observation_size=135, action_size=3)
    with pytest.raises(ValueError):
        build_ball_residual_actor_critic(observation_size=138, action_size=True)
