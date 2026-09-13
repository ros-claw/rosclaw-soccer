import copy

import pytest

from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo

torch = pytest.importorskip("torch")


class SeparatedLearner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Parameter(torch.zeros(32))
        self.critic = torch.nn.Linear(139, 1, bias=False)
        torch.nn.init.zeros_(self.critic.weight)
        self.logstd = torch.nn.Parameter(torch.full((32,), -3.5))

    def forward(self, obs):
        return self.actor.expand(len(obs), -1), self.critic(obs[:, :139]).squeeze(-1)


def case():
    model = SeparatedLearner()
    obs = torch.zeros(4, 2, 140)
    obs[:2, :, 139] = 1
    obs[2:, :, 0] = 1  # A feature seen only after the actor window closes.
    with torch.no_grad():
        mean, value = model(obs.reshape(-1, 140))
        normal = torch.distributions.Normal(mean, model.logstd.exp())
        raw = torch.full_like(mean, 0.01)
    rollout = dict(
        obs=obs,
        raw=raw.reshape(4, 2, 32),
        value=value.reshape(4, 2),
        logp=normal.log_prob(raw).sum(1).reshape(4, 2).detach(),
        reward=torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 2.0]]),
        alive=torch.ones(4, 2),
        next_alive=torch.ones(4, 2),
    )
    return model, rollout


def config(enabled):
    return FullBodyPPOUpdateConfig(
        epochs=2,
        minibatch_size=2,
        observation_size=140,
        action_size=32,
        minimum_log_std=-4,
        learning_observation_index=139,
        critic_all_active=enabled,
    )


def test_critic_learns_post_window_state_without_extending_actor_window():
    for enabled in (False, True):
        model, rollout = case()
        result = update_full_body_ppo(
            model, torch.optim.SGD(model.parameters(), lr=0.01), rollout, config(enabled)
        )
        assert result["active_samples"] == 4
        assert result["verified_active_samples"] == 8
        assert bool(model.critic.weight[0, 0] != 0) == enabled
        if enabled:
            assert result["critic_samples_per_epoch"] == 8


def test_post_window_action_sign_does_not_change_update_and_replay_is_exact():
    model, rollout = case()
    initial = copy.deepcopy(model.state_dict())
    results = []
    for change in (False, True):
        model.load_state_dict(initial)
        data = copy.deepcopy(rollout)
        if change:
            data["raw"][2:] *= -1  # Same Normal likelihood, different discarded actions.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        torch.manual_seed(1079)
        report = update_full_body_ppo(model, optimizer, data, config(True))
        results.append((copy.deepcopy(model.state_dict()), report))
    assert results[0][1] == results[1][1]
    for key in initial:
        torch.testing.assert_close(results[0][0][key], results[1][0][key], rtol=0, atol=0)


def test_post_window_likelihood_still_verified_before_update():
    model, rollout = case()
    before = copy.deepcopy(model.state_dict())
    rollout["logp"][3, 0] += 0.1
    with pytest.raises(ValueError, match="likelihood"):
        update_full_body_ppo(model, torch.optim.Adam(model.parameters()), rollout, config(True))
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_full_critic_does_not_learn_dead_padding():
    model, rollout = case()
    rollout["alive"][2:] = 0
    rollout["next_alive"][1:] = 0
    report = update_full_body_ppo(
        model, torch.optim.SGD(model.parameters(), lr=0.01), rollout, config(True)
    )
    assert report["critic_samples_per_epoch"] == 4
    assert float(model.critic.weight[0, 0].detach()) == 0


@pytest.mark.parametrize("invalid", [1, 0, None, "yes"])
def test_critic_scope_requires_explicit_bool(invalid):
    with pytest.raises(ValueError):
        FullBodyPPOUpdateConfig(critic_all_active=invalid)
