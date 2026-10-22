"""Exact AR(1) conditional motor-action likelihoods for research PPO.

Reference: Korenkevych et al., IJCAI2019, doi:10.24963/ijcai.2019/382.
Not a trained policy. No simulator, hidden-state transfer, action clipping or
device access. Rollout code must commit previous action/mean only after the
corresponding actual physical control, and recompute all means for updates.
"""

import math

import torch


def _check(mean: torch.Tensor, log_std: torch.Tensor, rho: float) -> None:
    if (
        mean.ndim not in (1, 2)
        or mean.shape[-1] != 29
        or (mean.ndim == 2 and not 1 <= len(mean) <= 512)
        or mean.dtype not in (torch.float32, torch.float64)
        or log_std.shape != (29,)
        or log_std.dtype != mean.dtype
        or log_std.device != mean.device
        or not torch.isfinite(mean).all()
        or not torch.isfinite(log_std).all()
        or type(rho) is not float
        or not 0 <= rho <= 0.95
    ):
        raise ValueError(
            "finite same-device motor29 means/scales and fixed bounded correlation required"
        )
    if ((log_std < math.log(0.005)) | (log_std > math.log(0.05))).any():
        raise ValueError("declared exploration standard deviation must stay in [.005,.05]")


def conditional_parameters(
    means: torch.Tensor, actions: torch.Tensor, log_std: torch.Tensor, *, rho: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact probability of actual action history under freshly recomputed means.

    Do not detach previous means: current likelihood depends on both mu[t]
    and mu[t-1]. The first innovation uses the stationary marginal sigma.
    """
    _check(means, log_std, rho)
    if (
        means.ndim != 2
        or actions.shape != means.shape
        or actions.dtype != means.dtype
        or actions.device != means.device
        or not torch.isfinite(actions).all()
    ):
        raise ValueError(
            "one complete unpadded actual action history aligned with own means required"
        )
    location = torch.cat((means[:1], means[1:] + rho * (actions[:-1] - means[:-1])))
    scale = torch.cat(
        (
            log_std.exp()[None],
            (log_std.exp() * math.sqrt(1 - rho * rho))[None].expand(len(means) - 1, -1),
        )
    )
    return location, scale


def conditional_logp(
    means: torch.Tensor, actions: torch.Tensor, log_std: torch.Tensor, *, rho: float
) -> torch.Tensor:
    location, scale = conditional_parameters(means, actions, log_std, rho=rho)
    return (
        -0.5 * ((actions - location) / scale).square() - scale.log() - 0.5 * math.log(2 * math.pi)
    ).sum(-1)


def sample_action(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    *,
    rho: float,
    previous_action: torch.Tensor | None,
    previous_mean: torch.Tensor | None,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stateless proposal; owns no execution/memory state or motor authority."""
    _check(mean, log_std, rho)
    if mean.shape != (29,) or (previous_action is None) != (previous_mean is None):
        raise ValueError("one own motor mean and paired actual previous action/mean required")
    first = previous_action is None
    location = mean
    if previous_action is not None:
        if previous_mean is None:
            raise ValueError("previous actual mean required")
        for value in (previous_action, previous_mean):
            if (
                value.shape != (29,)
                or value.dtype != mean.dtype
                or value.device != mean.device
                or not torch.isfinite(value).all()
            ):
                raise ValueError("finite own actual motor history required")
        location = mean + rho * (previous_action - previous_mean)
    scale = log_std.exp() * (1.0 if first else math.sqrt(1 - rho * rho))
    action = location + scale * torch.randn(
        mean.shape, generator=generator, dtype=mean.dtype, device=mean.device
    )
    logp = (
        -0.5 * ((action - location) / scale).square() - scale.log() - 0.5 * math.log(2 * math.pi)
    ).sum()
    return action, logp
