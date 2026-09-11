"""Redistribute an explicitly identified reward component without changing return.

This is accounting, not causal inference. The caller must justify and record
the earlier active action assigned to each delayed observation. Physics,
observations, actions and ongoing recovery masks remain untouched. No action
authority, policy promotion or optimal-policy guarantee is provided here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DelayedCreditReceipt:
    moved_components: int
    maximum_delay_steps: int
    discounted_return_max_error: float
    numeric_return_preserved: bool
    activation_ceiling: str = "SIM_ONLY"
    promotion_eligible: bool = False


def redistribute_delayed_component(
    reward: Any, component: Any, destination: Any, alive: Any, *, gamma: float
) -> tuple[Any, DelayedCreditReceipt]:
    """Move signed components to declared earlier actions, preserving discount.

    CPU tensors have shape (time, world), at most 65536 elements. ``destination``
    is int64, -1 exactly where the component is zero. Both source and destination
    must be active in the same non-restarting episode. A component moved from
    s to d is multiplied by gamma**(s-d). All unassigned reward stays in place.
    """
    import torch

    arrays = (reward, component, destination, alive)
    if (
        type(gamma) not in (float, int)
        or not math.isfinite(gamma)
        or not 0.9 <= gamma < 1
        or any(not isinstance(a, torch.Tensor) for a in arrays)
        or reward.ndim != 2
        or not all(1 <= n <= 4096 for n in reward.shape)
        or reward.numel() > 65536
        or reward.dtype not in (torch.float32, torch.float64)
        or component.dtype != reward.dtype
        or destination.dtype != torch.int64
        or alive.dtype not in (torch.bool, torch.float32, torch.float64)
        or any(a.shape != reward.shape or a.device.type != "cpu" for a in arrays)
        or any(a.layout != torch.strided or a.requires_grad for a in arrays)
        or not bool(torch.isfinite(reward).all() and torch.isfinite(component).all())
        or bool((reward.abs() > 1e6).any() or (component.abs() > 1e6).any())
        or not bool(((alive == 0) | (alive == 1)).all())
        or bool((alive[1:].to(torch.int8) > alive[:-1].to(torch.int8)).any())
    ):
        raise ValueError("bounded detached CPU episode/component tensors required")
    horizon, worlds = reward.shape
    time = torch.arange(horizon)[:, None].expand(horizon, worlds)
    event = component != 0
    if bool(
        ((~event) & (destination != -1)).any()
        or (event & ((destination < 0) | (destination > time))).any()
        or (event & (alive != 1)).any()
    ):
        raise ValueError("delayed components require same-episode active earlier destinations")
    source_t, world = torch.where(event)
    target_t = destination[source_t, world]
    if bool((alive[target_t, world] != 1).any()):
        raise ValueError("cannot assign outcome to an inactive action")
    moved = event & (destination < time)
    source_t, world = torch.where(moved)
    target_t = destination[source_t, world]
    result = reward.to(torch.float64).clone()
    if len(source_t):
        values = component[source_t, world].to(torch.float64)
        result[source_t, world] -= values
        transfer = values * gamma ** (source_t - target_t).to(torch.float64)
        result.view(-1).index_add_(0, target_t * worlds + world, transfer)
    result = result.to(reward.dtype)
    weights = gamma ** torch.arange(horizon, dtype=torch.float64)[:, None]
    old = reward.to(torch.float64)
    new = result.to(torch.float64)
    error = ((weights * old).sum(0) - (weights * new).sum(0)).abs()
    roundoff = 4 * torch.finfo(reward.dtype).eps * (weights * (old.abs() + new.abs())).sum(0)
    preserved = bool((error <= roundoff.clamp_min(1e-12)).all())
    if not preserved:
        raise FloatingPointError("discounted return changed beyond numeric roundoff")
    receipt = DelayedCreditReceipt(
        moved_components=len(source_t),
        maximum_delay_steps=int((source_t - target_t).max()) if len(source_t) else 0,
        discounted_return_max_error=float(error.max()),
        numeric_return_preserved=True,
    )
    return result, receipt
