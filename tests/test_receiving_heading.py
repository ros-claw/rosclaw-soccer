from dataclasses import replace

import numpy as np
import pytest
from test_s368_recurrent_receiver import CONFIG, POLICY, motor, observation

from rosclaw_soccer.providers.g1.recurrent_receiver import G1RecurrentReceiver
from rosclaw_soccer.sim.contracts import hash_bytes
from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic
from rosclaw_soccer.training.receiving_heading import migrate_receiving_heading

CONTRACT = "recurrent_receiver_world_heading_135_float32.v2"


def state(agent):
    return {k: v.detach().numpy().copy() for k, v in agent.state_dict().items()}


def test_explicit_extension_preserves_old_weights_and_resolves_heading_alias(tmp_path):
    parent = motor(tmp_path)
    raw = state(parent._actor)
    migrated = migrate_receiving_heading(raw)
    path = tmp_path / "heading.npz"
    np.savez_compressed(path, **migrated)
    actors = []
    for yaw in (0.0, np.pi):
        m = G1RecurrentReceiver(
            path,
            agent_id="blue.playmaker",
            expected_actor_hash=hash_bytes(path.read_bytes()),
            foundation_hash=POLICY,
            foundation_config_hash=CONFIG,
            observation_contract=CONTRACT,
        )
        o = observation()
        q = list(o.qpos)
        q[3] = float(np.cos(yaw / 2))
        q[6] = float(np.sin(yaw / 2))
        o = replace(o, qpos=tuple(q))
        m.begin_skill(o)
        m.propose(o)
        actors.append(m.last_observation.numpy())
    np.testing.assert_array_equal(actors[0][:, :133], actors[1][:, :133])
    assert actors[0][0, -1] == 1 and actors[1][0, -1] == -1
    assert parent.contract_hash != m.contract_hash
    for k, v in raw.items():
        np.testing.assert_array_equal(
            v, migrated[k][:, :133] if k.endswith(".0.weight") else migrated[k]
        )
    assert not np.any(migrated["actor.0.weight"][:, 133:])


def test_nonzero_migration_output_tolerance_and_heading_parameters_learn():
    import torch

    old = build_ball_residual_actor_critic()
    with torch.no_grad():
        old.actor[-1].weight.fill_(0.01)
    new = build_ball_residual_actor_critic(observation_size=135)
    new.load_state_dict(
        {k: torch.tensor(v) for k, v in migrate_receiving_heading(state(old)).items()}
    )
    x = torch.linspace(-1, 1, 4 * 133).reshape(4, 133)
    heading = torch.tensor([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]])
    extended = torch.cat((x, heading), 1)
    before = old(x)
    after = new(extended)
    for a, b in zip(before, after, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    new(extended)[0][0].sum().backward()
    assert torch.count_nonzero(new.actor[0].weight.grad[:, 133:]) > 0


def test_incompatible_or_damaged_migration_rejected():
    raw = state(build_ball_residual_actor_critic())
    raw["actor.0.weight"][0, 0] = np.nan
    with pytest.raises(ValueError):
        migrate_receiving_heading(raw)
    with pytest.raises(ValueError):
        build_ball_residual_actor_critic(observation_size=True)
    with pytest.raises(ValueError):
        build_ball_residual_actor_critic(observation_size=134)
