"""Box-truncated AR conditionals: sampling, normalized log-density, analytic KL.

No action clipping, rejection of a physical proposal, simulator or state import.
The conditional distribution is truncated; it is NOT a stationary Gaussian AR.
Bounds are conservative float32-representable points inside the original .2rad
reference-relative authority. CDF arithmetic uses float64 and stable log tails.

Primary distribution definition:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
"""

import math

import torch

from rosclaw_soccer.rsi.correlated_motor_exploration import conditional_parameters


def motor_box(
    reference: torch.Tensor, action_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        reference.ndim not in (1, 2)
        or reference.shape[-1] != 29
        or reference.dtype != torch.float32
        or action_scale.shape != (29,)
        or action_scale.dtype != reference.dtype
        or action_scale.device != reference.device
        or not torch.isfinite(reference).all()
        or not torch.isfinite(action_scale).all()
        or not (action_scale > 0).all()
    ):
        raise ValueError("finite actual reference29 and unchanged positive joint scales required")
    lower = (reference.double() - 0.2 / action_scale.double()).float()
    upper = (reference.double() + 0.2 / action_scale.double()).float()
    lower = torch.nextafter(lower, torch.full_like(lower, float("inf")))
    upper = torch.nextafter(upper, torch.full_like(upper, -float("inf")))
    if not (lower < upper).all():
        raise ValueError("nondegenerate representable motor box required")
    return lower.double(), upper.double()


def _terms(
    location: torch.Tensor, scale: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b = (
        (lower - location.double()) / scale.double(),
        (upper - location.double()) / scale.double(),
    )
    flipped = a > 0
    left, right = torch.where(flipped, -b, a), torch.where(flipped, -a, b)
    la, lb = torch.special.log_ndtr(left), torch.special.log_ndtr(right)
    logz = lb + torch.log(-torch.expm1(la - lb))
    if not torch.isfinite(logz).all():
        raise ValueError("finite stable truncated normal mass required")
    return a, b, logz, flipped, la


def bounded_ar_logp(
    means: torch.Tensor,
    actions: torch.Tensor,
    reference: torch.Tensor,
    action_scale: torch.Tensor,
    log_std: torch.Tensor,
    *,
    rho: float,
) -> torch.Tensor:
    if reference.shape != means.shape:
        raise ValueError("actual per-control authority reference required")
    location, scale = conditional_parameters(means, actions, log_std, rho=rho)
    lower, upper = motor_box(reference, action_scale)
    if ((actions.double() < lower) | (actions.double() > upper)).any():
        raise ValueError("actual bounded action outside declared probability support")
    _, _, logz, _, _ = _terms(location, scale, lower, upper)
    logp = (
        -0.5 * ((actions.double() - location.double()) / scale.double()).square()
        - scale.double().log()
        - 0.5 * math.log(2 * math.pi)
        - logz
    )
    return logp.sum(-1).to(means.dtype)


def bounded_ar_kl(
    old_means: torch.Tensor,
    new_means: torch.Tensor,
    actions: torch.Tensor,
    reference: torch.Tensor,
    action_scale: torch.Tensor,
    log_std: torch.Tensor,
    *,
    rho: float,
) -> torch.Tensor:
    old, scale = conditional_parameters(old_means, actions, log_std, rho=rho)
    new, _ = conditional_parameters(new_means, actions, log_std, rho=rho)
    lo, hi = motor_box(reference, action_scale)
    a, b, old_z, _, _ = _terms(old, scale, lo, hi)
    _, _, new_z, _, _ = _terms(new, scale, lo, hi)
    moment = scale.double() * (
        torch.exp(-0.5 * a * a - 0.5 * math.log(2 * math.pi) - old_z)
        - torch.exp(-0.5 * b * b - 0.5 * math.log(2 * math.pi) - old_z)
    )
    difference = old.double() - new.double()
    kl = (
        (
            new_z
            - old_z
            + (difference.square() + 2 * difference * moment) / (2 * scale.double().square())
        )
        .sum(-1)
        .mean()
    )
    if not torch.isfinite(kl) or kl < -1e-8:
        raise ValueError("finite nonnegative analytic truncated conditional KL required")
    return kl.clamp_min(0)


def _inverse_logcdf(logp: torch.Tensor) -> torch.Tensor:
    # Newton on log Phi remains usable when exp(logp) underflows in remote tails.
    tail = logp < -32
    x = torch.where(tail, -torch.sqrt(-2 * logp), torch.special.ndtri(logp.exp()))
    for _ in range(8):
        cdf = torch.special.log_ndtr(x)
        derivative = torch.exp(-0.5 * x * x - 0.5 * math.log(2 * math.pi) - cdf)
        x = torch.where(tail, x - (cdf - logp) / derivative, x)
    if not torch.isfinite(x).all():
        raise ValueError("finite inverse log-CDF sample required")
    return x


def sample_bounded_action(
    mean: torch.Tensor,
    reference: torch.Tensor,
    action_scale: torch.Tensor,
    log_std: torch.Tensor,
    *,
    rho: float,
    previous_action: torch.Tensor | None,
    previous_mean: torch.Tensor | None,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (previous_action is None) != (previous_mean is None) or mean.shape != (29,):
        raise ValueError("one actual motor mean and paired own previous history required")
    means = mean[None] if previous_mean is None else torch.stack([previous_mean, mean])
    actions = mean[None] if previous_action is None else torch.stack([previous_action, mean])
    locations, scales = conditional_parameters(means, actions, log_std, rho=rho)
    location, scale = locations[-1], scales[-1]
    lower, upper = motor_box(reference, action_scale)
    _, _, logz, flipped, la = _terms(location, scale, lower, upper)
    u = torch.rand((29,), dtype=torch.float64, device=mean.device, generator=generator)
    quantile = _inverse_logcdf(torch.logaddexp(la, u.log() + logz))
    z = torch.where(flipped, -quantile, quantile)
    action = (location.double() + scale.double() * z).to(mean.dtype)
    if ((action.double() < lower) | (action.double() > upper)).any():
        raise ValueError("finite-precision inverse-CDF escaped probability support; do not execute")
    logp = (
        (
            -0.5 * ((action.double() - location.double()) / scale.double()).square()
            - scale.double().log()
            - 0.5 * math.log(2 * math.pi)
            - logz
        )
        .sum()
        .to(mean.dtype)
    )
    return action, logp
