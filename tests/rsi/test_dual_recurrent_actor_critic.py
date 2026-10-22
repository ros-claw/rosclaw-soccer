import math

import pytest
import torch
from test_dual_recurrent_motor import model

from rosclaw_soccer.rsi.correlated_motor_exploration import conditional_logp
from rosclaw_soccer.rsi.dual_recurrent_actor_critic import (
    MotorCritic,
    own_history_conditional_kl,
    own_history_loss,
)


def fixture():
    _, actor = model()
    critic = MotorCritic()
    obs = torch.randn(12, 160) * 0.1
    with torch.no_grad():
        means, _ = actor(obs[:, None], actor.initial_state())
    means = means[:, 0]
    std = torch.full((29,), math.log(0.01))
    actions = means + 0.002 * torch.randn_like(means)
    row = dict(
        observations=obs,
        actions=actions,
        old_means=means,
        old_logp=conditional_logp(means, actions, std, rho=0.9),
        old_values=torch.zeros(12),
        returns=torch.ones(12),
        advantages=torch.linspace(-1, 1, 12),
        assisted=False,
        all_actor_actions_actually_executed=True,
        authority_rejected=False,
        natural_terminal_observed=True,
    )
    return actor, critic, row, std


def test_own_history_identity_kl_zero_and_actor_critic_both_get_gradients():
    actor, critic, row, std = fixture()
    loss, info = own_history_loss(actor, critic, row, std, rho=0.9)
    # CPU LSTM no-grad and autograd kernels need not be bitwise identical.
    assert 0 <= info["conditional_kl"] < 1e-8
    loss.backward()
    assert all(p.grad is None for p in actor.parent.parameters())
    assert actor.plastic.rnn.weight_ih_l0.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in critic.parameters())


def test_value_only_cannot_silently_update_the_motor():
    actor, critic, row, std = fixture()
    loss, info = own_history_loss(actor, critic, row, std, rho=0.9, value_only=True)
    loss.backward()
    assert info["actor_rows"] == 0
    assert all(p.grad is None for p in actor.plastic.parameters())


@pytest.mark.parametrize(
    "field,value",
    [
        ("assisted", True),
        ("all_actor_actions_actually_executed", False),
        ("authority_rejected", True),
        ("natural_terminal_observed", False),
    ],
)
def test_never_credit_teacher_rescue_or_unexecuted_action(field, value):
    actor, critic, row, std = fixture()
    row[field] = value
    with pytest.raises(ValueError):
        own_history_loss(actor, critic, row, std, rho=0.9)


def test_wrong_sampling_likelihood_rejected():
    actor, critic, row, std = fixture()
    row["old_logp"] = conditional_logp(row["old_means"], row["actions"], std, rho=0.0)
    with pytest.raises(ValueError):
        own_history_loss(actor, critic, row, std, rho=0.9)


def test_large_trial_kl_backtracks_without_relaxing_the_likelihood_guard():
    actor, critic, row, std = fixture()
    with torch.no_grad():
        actor.plastic.head.bias.add_(10)
    kl = own_history_conditional_kl(actor, row, std, rho=0.9)
    assert math.isfinite(kl) and kl > 0.015
    with pytest.raises(ValueError, match="invalid whole-history AR likelihood ratio"):
        own_history_loss(actor, critic, row, std, rho=0.9)


@pytest.mark.parametrize("sigma", [0.005, 0.02])
def test_bounded_ppo_uses_actual_authority_and_normalized_sampling_density(sigma):
    from rosclaw_soccer.rsi.motor_sampling_law import history_motor_logp

    actor, critic, row, std = fixture()
    std = torch.full_like(std, math.log(sigma))
    with torch.no_grad():
        actor.plastic.head.bias.add_(0.399)
        mean, _, e = actor.forward_with_evidence(
            row["observations"][:, None], actor.initial_state()
        )
    row.update(
        sampling_law="box-truncated-ar1",
        old_means=mean[:, 0],
        actions=mean[:, 0] - 0.002,
        reference_nominal=e["reference_nominal"][:, 0],
        action_scale=actor.parent.get_buffer("action_scale"),
    )
    row["old_logp"] = history_motor_logp(
        row["sampling_law"],
        row["old_means"],
        row["actions"],
        row["reference_nominal"],
        row["action_scale"],
        std,
        rho=0.9,
    )
    loss, info = own_history_loss(actor, critic, row, std, rho=0.9)
    assert torch.isfinite(loss) and info["conditional_kl"] < 1e-8
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.plastic.parameters())
    row["old_logp"] = conditional_logp(row["old_means"], row["actions"], std, rho=0.9)
    with pytest.raises(ValueError, match="old AR likelihood"):
        own_history_loss(actor, critic, row, std, rho=0.9)
