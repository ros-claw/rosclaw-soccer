"""Export learned output offsets into a cloned actor, without runtime hooks.

This is parameter packaging, not proof of learning quality or execution
authority. Callers must re-evaluate the exported policy: floating-point fusion
need not be bit-identical to adding an offset after inference.
"""

from __future__ import annotations

import copy
import math
from typing import Any


def fuse_actor_output_offset(
    agent: Any, offset: Any, *, maximum_absolute_offset: float = 2.0
) -> Any:
    """Clone a sequential actor-critic and adjust only its final linear bias.

    Accepts a detached, finite offset in the existing bias dtype and device.
    The original actor, critic, buffers and optimizer-owned parameters are
    never mutated. The returned model grants no policy activation authority.
    """
    import torch

    actor: Any = getattr(agent, "actor", None)
    if (
        not isinstance(agent, torch.nn.Module)
        or type(actor) is not torch.nn.Sequential
        or len(actor) == 0
        or type(actor[-1]) is not torch.nn.Linear
        or actor[-1].bias is None
    ):
        raise ValueError("a sequential actor with final biased linear layer is required")
    bias = actor[-1].bias
    if (
        type(maximum_absolute_offset) not in (int, float)
        or not math.isfinite(maximum_absolute_offset)
        or not 0 < maximum_absolute_offset <= 2.0
        or not isinstance(offset, torch.Tensor)
        or offset.requires_grad
        or offset.ndim != 1
        or offset.shape != bias.shape
        or not 1 <= offset.numel() <= 256
        or offset.dtype not in (torch.float32, torch.float64)
        or offset.dtype != bias.dtype
        or offset.device != bias.device
        or not bool(torch.isfinite(offset).all())
        or bool((offset.abs() > maximum_absolute_offset).any())
    ):
        raise ValueError("a detached finite bounded offset matching the actor bias is required")
    if any(not bool(torch.isfinite(value).all()) for value in agent.state_dict().values()):
        raise ValueError("source policy parameters and buffers must be finite")
    candidate = copy.deepcopy(agent)
    candidate_actor: Any = candidate.actor
    with torch.no_grad():
        candidate_actor[-1].bias.add_(offset)
    if not bool(torch.isfinite(candidate_actor[-1].bias).all()):
        raise ValueError("fused actor bias overflow")
    return candidate
