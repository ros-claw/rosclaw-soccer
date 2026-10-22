import math

import numpy as np
import pytest
import torch
from scipy.special import ndtri_exp
from scipy.stats import truncnorm

from rosclaw_soccer.rsi.bounded_motor_exploration import (
    _inverse_logcdf,
    bounded_ar_kl,
    bounded_ar_logp,
    motor_box,
    sample_bounded_action,
)
from rosclaw_soccer.rsi.correlated_motor_exploration import conditional_parameters


def fixture():
    return torch.zeros(29), torch.full((29,), 0.5), torch.full((29,), math.log(0.005))


def test_inverse_logcdf_agrees_with_primary_numerical_reference_in_deep_tails():
    logp = torch.tensor([-1.0, -40.0, -900.0, -5000.0], dtype=torch.float64)
    np.testing.assert_allclose(
        _inverse_logcdf(logp).numpy(), ndtri_exp(logp.numpy()), atol=1e-11, rtol=1e-12
    )


def test_log_density_includes_boundary_mass_and_agrees_with_scipy():
    ref, scale, std = fixture()
    means = torch.full((3, 29), 0.4)
    actions = means - 0.002
    references = ref.expand(3, -1)
    loc, sigma = conditional_parameters(means, actions, std, rho=0.9)
    lo, hi = motor_box(references, scale)
    actual = bounded_ar_logp(means, actions, references, scale, std, rho=0.9)
    expected = truncnorm.logpdf(
        actions.numpy(),
        (lo.numpy() - loc.numpy()) / sigma.numpy(),
        (hi.numpy() - loc.numpy()) / sigma.numpy(),
        loc=loc.numpy(),
        scale=sigma.numpy(),
    ).sum(-1)
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-5, rtol=1e-6)


def test_samples_stay_inside_authority_without_clipping_and_have_half_normal_mean():
    ref, scale, std = fixture()
    rng = torch.Generator().manual_seed(15)
    values = []
    for _ in range(300):
        action, lp = sample_bounded_action(
            torch.full((29,), 0.4),
            ref,
            scale,
            std,
            rho=0.9,
            previous_action=None,
            previous_mean=None,
            generator=rng,
        )
        assert (((action - ref) * scale).abs() <= 0.2).all()
        assert torch.isfinite(lp)
        values.append(action[0].item())
    assert np.mean(values) == pytest.approx(0.4 - 0.005 * math.sqrt(2 / math.pi), abs=0.001)


def test_online_sampling_and_whole_history_density_match():
    ref, scale, std = fixture()
    rng = torch.Generator().manual_seed(41)
    previous = oldmean = None
    means = []
    actions = []
    probabilities = []
    for i in range(12):
        mean = torch.full((29,), 0.4 if i % 3 else -0.4)
        action, lp = sample_bounded_action(
            mean,
            ref,
            scale,
            std,
            rho=0.9,
            previous_action=previous,
            previous_mean=oldmean,
            generator=rng,
        )
        means.append(mean)
        actions.append(action)
        probabilities.append(lp)
        previous, oldmean = action, mean
    actual = bounded_ar_logp(
        torch.stack(means), torch.stack(actions), ref.expand(12, -1), scale, std, rho=0.9
    )
    torch.testing.assert_close(actual, torch.stack(probabilities), atol=1e-4, rtol=1e-6)


def test_conditional_kl_agrees_with_numerical_integration():
    ref, scale, std = fixture()
    old = torch.full((1, 29), 0.399)
    new = torch.full((1, 29), 0.397)
    actions = old - 0.002
    refs = ref[None]
    kl = bounded_ar_kl(old, new, actions, refs, scale, std, rho=0.9)
    lo, hi = motor_box(ref, scale)
    lower, upper = lo[0].item(), hi[0].item()
    sigma = std.exp()[0].item()
    p = truncnorm(
        (lower - old[0, 0].item()) / sigma,
        (upper - old[0, 0].item()) / sigma,
        loc=old[0, 0].item(),
        scale=sigma,
    )
    q = truncnorm(
        (lower - new[0, 0].item()) / sigma,
        (upper - new[0, 0].item()) / sigma,
        loc=new[0, 0].item(),
        scale=sigma,
    )
    expected = 29 * p.expect(lambda x: p.logpdf(x) - q.logpdf(x))
    assert float(kl) == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert bounded_ar_kl(old, old, actions, refs, scale, std, rho=0.9) == 0


def test_probability_gradient_keeps_previous_mean_and_truncation_normalizer():
    ref, scale, std = fixture()
    means = torch.full((2, 29), 0.399, requires_grad=True)
    actions = means.detach() - 0.003
    lp = bounded_ar_logp(means, actions, ref.expand(2, -1), scale, std, rho=0.9)
    lp[1].backward()
    assert means.grad[0].abs().sum() > 0 and means.grad[1].abs().sum() > 0
    with pytest.raises(ValueError):
        bounded_ar_logp(means, torch.full((2, 29), 0.5), ref.expand(2, -1), scale, std, rho=0.9)
