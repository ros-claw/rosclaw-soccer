import numpy as np
import pytest
import torch
from test_paid_trainable_actor import synthetic_actor

from rosclaw_soccer.rsi.paid_recurrent_ppo import (
    GAMMA,
    backtracked_adam_update,
    finite_option_gae,
    recurrent_episode_loss,
)


def test_true_finite_option_terminal_and_real_tail_discount():
    v = np.array([1, 2, 3], np.float32)
    adv, ret = finite_option_gae(v, 10, 60)
    assert ret[-1] == pytest.approx(10 * GAMMA**60)
    assert adv[-1] == pytest.approx(10 * GAMMA**60 - 3)
    assert np.isfinite(ret).all()
    with pytest.raises(ValueError):
        finite_option_gae(v, 1, 61)


def test_monte_carlo_preserves_real_reward_delay_without_lambda_suppression():
    value = np.linspace(-1, 2, 289, dtype=np.float32)
    adv, returns = finite_option_gae(value, 10, 60, credit_mode="monte-carlo")
    assert returns[104] == pytest.approx(10 * GAMMA ** (288 - 104 + 60))
    assert returns[-1] == pytest.approx(10 * GAMMA**60)
    np.testing.assert_array_equal(adv, returns - value)
    other, other_returns = finite_option_gae(
        np.zeros_like(value), 10, 60, credit_mode="monte-carlo"
    )
    np.testing.assert_array_equal(returns, other_returns)
    np.testing.assert_array_equal(other, returns)
    with pytest.raises(ValueError, match="credit rule"):
        finite_option_gae(value, 10, 0, credit_mode="hindsight-shifted-reward")


def test_recurrent_loss_recomputes_prefix_and_backpropagates_to_rnn():
    import copy

    torch.manual_seed(71)
    actor = synthetic_actor()
    anchor = copy.deepcopy(actor).eval()
    critic = torch.nn.Linear(160, 1)
    std = torch.nn.Parameter(torch.full((29,), float(np.log(0.02))))
    obs = torch.randn(5, 160)
    with torch.no_grad():
        mean, _ = actor(obs[:, None], (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128)))
        d = torch.distributions.Normal(mean[:, 0], std.exp())
        action = d.sample()
        episode = dict(
            obs=obs,
            action=action,
            logp=d.log_prob(action).sum(-1),
            value=critic(obs).squeeze(-1),
            returns=torch.ones(5),
            advantage=torch.arange(5, dtype=torch.float32) - 2,
        )
    loss, stats = recurrent_episode_loss(actor, anchor, critic, std, episode, value_only=False)
    assert float(stats["approximate_kl"]) == pytest.approx(0, abs=1e-7)
    assert stats["actor_rows"] == 5 and stats["anchor_rows"] == 5
    loss.backward()
    assert actor.rnn.weight_hh_l0.grad is not None
    assert actor.rnn.weight_hh_l0.grad.abs().sum() > 0
    assert not any(p.grad is not None for p in anchor.parameters())
    actor.zero_grad(set_to_none=True)
    loss, stats = recurrent_episode_loss(actor, anchor, critic, std, episode, value_only=True)
    loss.backward()
    assert not any(p.grad is not None for p in actor.parameters())
    assert stats["actor_rows"] == 0 and stats["anchor_rows"] == 0


def test_post_update_backtracking_commits_one_adam_step_not_all_trials():
    p = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.Adam([p], lr=1.0)
    p.grad = torch.ones_like(p)
    result = backtracked_adam_update(opt, [p], lambda: float(p.detach().square()), maximum_kl=0.01)
    assert result["applied"] and len(result["attempts"]) > 1
    assert float(p.detach().square()) <= 0.01
    assert float(opt.state[p]["step"]) == 1
    assert opt.param_groups[0]["lr"] < 1.0


def test_rejected_or_faulting_search_restores_weights_moments_and_learning_rate():
    p = torch.nn.Parameter(torch.zeros(()))
    opt = torch.optim.Adam([p], lr=1.0)
    p.grad = torch.ones_like(p)
    result = backtracked_adam_update(opt, [p], lambda: 1.0, maximum_attempts=2)
    assert not result["applied"] and float(p.detach()) == 0
    assert not opt.state and opt.param_groups[0]["lr"] == 1.0

    def broken():
        raise RuntimeError("broken comparison")

    with pytest.raises(RuntimeError, match="broken"):
        backtracked_adam_update(opt, [p], broken)
    assert float(p.detach()) == 0 and not opt.state
