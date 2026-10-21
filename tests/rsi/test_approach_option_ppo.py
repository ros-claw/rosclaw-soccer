import torch
from test_native_contact_adapter import Base

from rosclaw_soccer.rsi.approach_option_ppo import approach_ppo_loss, approach_sample
from rosclaw_soccer.rsi.native_contact_adapter import NativeResidualContactActor
from rosclaw_soccer.rsi.native_contact_ppo import make_contact_actor_critic


def setup():
    torch.manual_seed(7608)
    policy = make_contact_actor_critic(NativeResidualContactActor(Base()))
    obs = torch.randn(16, 177)
    obs[:8, 168:170] = torch.tensor([1.0, 0.0])
    obs[8:, 168:170] = torch.tensor([0.2, 0.0])
    with torch.no_grad():
        data = dict(obs=obs, **approach_sample(policy, obs))
    data.update(advantage=torch.linspace(-1, 1, 16), returns=data["old_value"] + 2)
    return policy, data


def test_sampled_far_actions_are_actual_immutable_mean_not_gaussian_noise():
    policy, data = setup()
    assert not data["plastic"][:8].any()
    assert data["plastic"][8:].all()
    with torch.no_grad():
        assert torch.equal(data["action"][:8], policy.actor.base(data["obs"])[:8])
    assert not data["old_logp"][:8].any()


def test_real_optimizer_updates_only_adapter_critic_and_exploration():
    policy, data = setup()
    base = {k: v.clone() for k, v in policy.actor.base.state_dict().items()}
    adapter = {k: v.clone() for k, v in policy.actor.adapter.state_dict().items()}
    opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=0.001)
    loss, stats = approach_ppo_loss(policy, data, value_only=False)
    assert stats["plastic_rows"] == 8 and stats["approximate_kl"].abs() < 1e-6
    loss.backward()
    opt.step()
    assert any(not torch.equal(v, adapter[k]) for k, v in policy.actor.adapter.state_dict().items())
    assert all(torch.equal(v, base[k]) for k, v in policy.actor.base.state_dict().items())
    assert all(p.grad is None and p not in opt.state for p in policy.actor.base.parameters())


def test_value_only_cannot_update_actor_or_exploration():
    policy, data = setup()
    before = {k: v.clone() for k, v in policy.state_dict().items() if not k.startswith("critic.")}
    opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=0.001)
    loss, _ = approach_ppo_loss(policy, data, value_only=True)
    loss.backward()
    opt.step()
    assert all(torch.equal(v, policy.state_dict()[k]) for k, v in before.items())
