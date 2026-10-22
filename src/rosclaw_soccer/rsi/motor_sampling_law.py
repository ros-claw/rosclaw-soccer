"""Explicit probability-law dispatch; legacy Gaussian evidence never changes law."""

import torch

from rosclaw_soccer.rsi.bounded_motor_exploration import (
    bounded_ar_kl,
    bounded_ar_logp,
    sample_bounded_action,
)
from rosclaw_soccer.rsi.correlated_motor_exploration import (
    conditional_logp,
    conditional_parameters,
    sample_action,
)


def draw_motor_action(
    law: str,
    mean: torch.Tensor,
    reference: torch.Tensor,
    scale: torch.Tensor,
    log_std: torch.Tensor,
    *,
    rho: float,
    previous_action: torch.Tensor | None,
    previous_mean: torch.Tensor | None,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if law == "box-truncated-ar1":
        return sample_bounded_action(
            mean,
            reference,
            scale,
            log_std,
            rho=rho,
            previous_action=previous_action,
            previous_mean=previous_mean,
            generator=generator,
        )
    if law != "gaussian-ar1":
        raise ValueError("explicit admitted motor probability law required")
    return sample_action(
        mean,
        log_std,
        rho=rho,
        previous_action=previous_action,
        previous_mean=previous_mean,
        generator=generator,
    )


def history_motor_logp(
    law: str,
    means: torch.Tensor,
    actions: torch.Tensor,
    reference: torch.Tensor | None,
    scale: torch.Tensor | None,
    log_std: torch.Tensor,
    *,
    rho: float,
) -> torch.Tensor:
    if law == "gaussian-ar1":
        return conditional_logp(means, actions, log_std, rho=rho)
    if law != "box-truncated-ar1" or reference is None or scale is None:
        raise ValueError("bounded law requires actual admitted reference/authority history")
    return bounded_ar_logp(means, actions, reference, scale, log_std, rho=rho)


def history_motor_kl(
    law: str,
    old_means: torch.Tensor,
    new_means: torch.Tensor,
    actions: torch.Tensor,
    reference: torch.Tensor | None,
    scale: torch.Tensor | None,
    log_std: torch.Tensor,
    *,
    rho: float,
) -> torch.Tensor:
    if law == "box-truncated-ar1":
        if reference is None or scale is None:
            raise ValueError("actual authority box required")
        return bounded_ar_kl(old_means, new_means, actions, reference, scale, log_std, rho=rho)
    if law != "gaussian-ar1":
        raise ValueError("explicit admitted motor probability law required")
    old, std = conditional_parameters(old_means, actions, log_std, rho=rho)
    new, _ = conditional_parameters(new_means, actions, log_std, rho=rho)
    return 0.5 * ((new - old) / std).square().sum(-1).mean()
