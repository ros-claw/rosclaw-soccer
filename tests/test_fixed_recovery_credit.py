import pytest

from rosclaw_soccer.training.fixed_recovery_credit import fold_fixed_recovery_tail


def data():
    torch = pytest.importorskip("torch")
    return torch, dict(
        obs=torch.zeros(4, 1, 143),
        raw=torch.zeros(4, 1, 29),
        logp=torch.zeros(4, 1),
        value=torch.zeros(4, 1),
        reward=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        alive=torch.ones(4, 1),
        next_alive=torch.tensor([[1.0], [1.0], [1.0], [0.0]]),
    )


def test_discounted_return_preserved_without_mutating_source():
    torch, source = data()
    original = {k: v.clone() for k, v in source.items()}
    folded = fold_fixed_recovery_tail(source, prefix_frames=2, gamma=0.95)
    assert folded["reward"][1].item() == pytest.approx(2 + 0.95 * 3 + 0.95**2 * 4)
    assert folded["next_alive"][-1].item() == 0
    assert all(torch.equal(v, original[k]) for k, v in source.items())


def test_death_in_recovery_stops_credit():
    _, source = data()
    source["next_alive"][2] = 0
    source["alive"][3] = 0
    source["reward"][3] = 0
    assert fold_fixed_recovery_tail(source, prefix_frames=2, gamma=0.95)["reward"][
        1
    ].item() == pytest.approx(2 + 0.95 * 3)


def test_folded_prefix_is_accepted_by_on_policy_optimizer():
    torch = pytest.importorskip("torch")
    from rosclaw_soccer.training.full_body_ppo import (
        FullBodyPPOUpdateConfig,
        update_full_body_ppo,
    )
    from rosclaw_soccer.training.handoff_policy import build_handoff_actor_critic

    torch.manual_seed(566)
    actor = build_handoff_actor_critic(action_size=29)
    optimizer = torch.optim.Adam(actor.parameters(), lr=5e-6)
    obs = torch.randn(4, 2, 143) * 0.1
    with torch.no_grad():
        mean, value = actor(obs.flatten(0, 1))
        normal = torch.distributions.Normal(mean, actor.logstd.clamp(-5.5, -0.3).exp())
        raw = normal.sample()
        source = dict(
            obs=obs,
            raw=raw.reshape(4, 2, 29),
            value=value.reshape(4, 2),
            logp=normal.log_prob(raw).sum(1).reshape(4, 2),
            reward=torch.randn(4, 2),
            alive=torch.ones(4, 2),
            next_alive=torch.ones(4, 2),
        )
    batch = fold_fixed_recovery_tail(source, prefix_frames=2, gamma=0.995)
    result = update_full_body_ppo(
        actor,
        optimizer,
        batch,
        FullBodyPPOUpdateConfig(
            observation_size=143,
            action_size=29,
            minimum_log_std=-5.5,
            epochs=1,
            minibatch_size=4,
        ),
    )
    assert result["optimizer_steps"] == 1
    assert result["on_policy_inputs_verified"]
    assert not result["promotion_eligible"]


def test_death_before_recovery_cannot_receive_tail_credit():
    _, source = data()
    source["next_alive"][1:] = 0
    source["alive"][2:] = 0
    source["reward"][2:] = 0
    folded = fold_fixed_recovery_tail(source, prefix_frames=2, gamma=0.95)
    assert folded["reward"][1].item() == 2


def test_no_suffix_retains_rewards_and_closes_horizon():
    torch, source = data()
    source["next_alive"][-1] = 1
    folded = fold_fixed_recovery_tail(source, prefix_frames=4, gamma=0.95)
    assert torch.equal(folded["reward"], source["reward"])
    assert folded["next_alive"][-1].item() == 0
    assert source["next_alive"][-1].item() == 1


def test_finite_rewards_cannot_overflow_return_silently():
    torch, source = data()
    source["reward"].fill_(torch.finfo(torch.float32).max)
    with pytest.raises(ValueError, match="overflow"):
        fold_fixed_recovery_tail(source, prefix_frames=2, gamma=0.95)


@pytest.mark.parametrize("fault", ["revive", "dead_reward", "nan", "prefix", "gamma", "missing"])
def test_invalid_tail_rejected(fault):
    _, source = data()
    prefix = 2
    gamma = 0.95
    if fault == "revive":
        source["alive"][2] = 0
    elif fault == "dead_reward":
        source["alive"][3] = 0
        source["next_alive"][2] = 0
    elif fault == "nan":
        source["reward"][2] = float("nan")
    elif fault == "prefix":
        prefix = True
    elif fault == "gamma":
        gamma = 1.0
    else:
        del source["raw"]
    with pytest.raises(ValueError):
        fold_fixed_recovery_tail(source, prefix_frames=prefix, gamma=gamma)
