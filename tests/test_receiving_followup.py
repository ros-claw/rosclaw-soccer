import numpy as np
import pytest
from test_s368_recurrent_receiver import CONFIG, POLICY, observation

from rosclaw_soccer.providers.g1.recurrent_receiver import G1RecurrentReceiver
from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic
from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig
from rosclaw_soccer.training.receiving_followup import migrate_receiving_followup

CONTRACT = "recurrent_receiver_followup_target_138_float32.v3"


def state(actor):
    return {k: v.detach().numpy().copy() for k, v in actor.state_dict().items()}


def test_migration_preserves_weights_and_can_learn_target_features():
    import torch

    parent = build_ball_residual_actor_critic(observation_size=135)
    with torch.no_grad():
        parent.actor[-1].weight.fill_(0.01)
    raw = state(parent)
    migrated = migrate_receiving_followup(raw)
    for k, v in raw.items():
        np.testing.assert_array_equal(
            v, migrated[k][:, :135] if k.endswith(".0.weight") else migrated[k]
        )
    child = build_ball_residual_actor_critic(observation_size=138)
    child.load_state_dict({k: torch.tensor(v) for k, v in migrated.items()})
    x = torch.linspace(-1, 1, 135)[None]
    y = torch.cat((x, torch.tensor([[1.0, -1.0, 0.2]])), 1)
    for a, b in zip(parent(x), child(y), strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    child(y)[0].sum().backward()
    assert torch.count_nonzero(child.actor[0].weight.grad[:, 135:]) > 0
    assert FullBodyPPOUpdateConfig(observation_size=138).observation_size == 138


def test_explicit_followup_target_changes_observation_and_identity(tmp_path):
    raw = state(build_ball_residual_actor_critic(observation_size=135))
    path = tmp_path / "followup.npz"
    np.savez_compressed(path, **migrate_receiving_followup(raw))
    observations, hashes = [], []
    for target in ((-1.5, -0.8, 1.5), (7.5, 0.8, 1.5)):
        backend = G1RecurrentReceiver(
            path,
            agent_id="blue.playmaker",
            expected_actor_hash=hash_bytes(path.read_bytes()),
            foundation_hash=POLICY,
            foundation_config_hash=CONFIG,
            observation_contract=CONTRACT,
            followup_target_position_m=target,
        )
        value = observation()
        backend.begin_skill(value)
        backend.propose(value)
        observations.append(backend.last_observation.numpy())
        hashes.append(backend.contract_hash)
        np.testing.assert_allclose(
            observations[-1][0, -3:],
            (np.asarray(target) - np.asarray(value.qpos[:3])) / 5,
            atol=1e-7,
        )
    np.testing.assert_array_equal(observations[0][:, :135], observations[1][:, :135])
    assert hashes[0] != hashes[1]


@pytest.mark.parametrize(
    "target", [None, (0.0, 0.0), (float("nan"), 0.0, 0.0), (201.0, 0.0, 0.0), [0.0, 0.0, 0.0]]
)
def test_missing_or_mutable_or_unbounded_target_rejected(tmp_path, target):
    with pytest.raises(ValueError, match="next-skill target"):
        G1RecurrentReceiver(
            tmp_path / "not-opened.npz",
            agent_id="blue.playmaker",
            expected_actor_hash=POLICY,
            foundation_hash=POLICY,
            foundation_config_hash=CONFIG,
            observation_contract=CONTRACT,
            followup_target_position_m=target,
        )


def test_bad_migration_rejected():
    raw = state(build_ball_residual_actor_critic(observation_size=135))
    raw["actor.0.weight"][0, 0] = np.nan
    with pytest.raises(ValueError):
        migrate_receiving_followup(raw)
