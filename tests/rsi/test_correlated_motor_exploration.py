import math

import pytest
import torch

from rosclaw_soccer.rsi.correlated_motor_exploration import conditional_logp, sample_action


def test_joint_density_equals_exact_multivariate_gaussian():
    generator = torch.Generator().manual_seed(45)
    means = torch.randn(7, 29, dtype=torch.float64, generator=generator) * 0.1
    actions = means + torch.randn(7, 29, dtype=torch.float64, generator=generator) * 0.02
    log_std = torch.full((29,), math.log(0.02), dtype=torch.float64)
    rho = 0.8
    index = torch.arange(7, dtype=torch.float64)
    covariance = 0.02**2 * rho ** (index[:, None] - index[None, :]).abs()
    expected = (
        torch.distributions.MultivariateNormal(means.T, covariance_matrix=covariance)
        .log_prob(actions.T)
        .sum()
    )
    assert torch.allclose(
        conditional_logp(means, actions, log_std, rho=rho).sum(), expected, atol=1e-9, rtol=0
    )


def test_white_noise_special_case():
    means = torch.zeros(3, 29)
    actions = torch.full_like(means, 0.01)
    log_std = torch.full((29,), math.log(0.02))
    expected = torch.distributions.Normal(means, log_std.exp()).log_prob(actions).sum(-1)
    assert torch.allclose(conditional_logp(means, actions, log_std, rho=0.0), expected)


def test_previous_mean_gradient_must_not_be_detached():
    means = torch.zeros(2, 29, dtype=torch.float64, requires_grad=True)
    actions = torch.zeros_like(means)
    actions[1] = 0.01
    log_std = torch.full((29,), math.log(0.02), dtype=torch.float64)
    conditional_logp(means, actions, log_std, rho=0.8)[1].backward()
    assert torch.all(means.grad[0] < 0)
    assert torch.allclose(means.grad[0], -0.8 * means.grad[1])


def test_online_sampling_matches_full_history_log_probability():
    generator = torch.Generator().manual_seed(27)
    means = torch.linspace(-0.03, 0.03, 6)[:, None].expand(-1, 29).clone()
    log_std = torch.full((29,), math.log(0.02))
    actions = []
    logps = []
    previous_action = previous_mean = None
    for mean in means:
        action, logp = sample_action(
            mean,
            log_std,
            rho=0.9,
            previous_action=previous_action,
            previous_mean=previous_mean,
            generator=generator,
        )
        actions.append(action)
        logps.append(logp)
        previous_action, previous_mean = action, mean
    assert torch.allclose(
        torch.stack(logps),
        conditional_logp(means, torch.stack(actions), log_std, rho=0.9),
        atol=1e-5,
    )


@pytest.mark.parametrize("rho", [-0.1, 1.0, float("nan")])
def test_bad_correlation_rejected(rho):
    with pytest.raises(ValueError):
        conditional_logp(
            torch.zeros(2, 29), torch.zeros(2, 29), torch.full((29,), math.log(0.02)), rho=rho
        )


def test_missing_previous_control_rejected():
    with pytest.raises(ValueError):
        sample_action(
            torch.zeros(29),
            torch.full((29,), math.log(0.02)),
            rho=0.9,
            previous_action=torch.zeros(29),
            previous_mean=None,
            generator=torch.Generator(),
        )
