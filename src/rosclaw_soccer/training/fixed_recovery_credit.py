"""Fold an explicitly non-policy recovery suffix into the final policy action.

The collector must guarantee that suffix actions do not depend on new sampled
actor outputs. This helper cannot verify control ownership and grants none.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def fold_fixed_recovery_tail(
    rollout: Mapping[str, Any], *, prefix_frames: int, gamma: float
) -> dict[str, Any]:
    import torch

    if set(rollout) != {"obs", "raw", "logp", "value", "reward", "alive", "next_alive"}:
        raise ValueError("complete fixed-recovery rollout required")
    reward = rollout["reward"]
    if (
        not isinstance(reward, torch.Tensor)
        or reward.ndim != 2
        or not 1 <= reward.shape[0] <= 4096
        or not 1 <= reward.shape[1] <= 4096
        or type(prefix_frames) is not int
        or not 1 <= prefix_frames <= reward.shape[0]
        or type(gamma) not in (float, int)
        or not math.isfinite(gamma)
        or not 0.9 <= gamma < 1
    ):
        raise ValueError("bounded fixed-recovery horizon and discount required")
    for name, value in rollout.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != (3 if name in ("obs", "raw") else 2)
            or value.shape[:2] != reward.shape
            or value.device != reward.device
            or value.layout != torch.strided
            or value.requires_grad
            or value.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("aligned detached fixed-recovery rollout required")
    alive, next_alive = rollout["alive"], rollout["next_alive"]
    if (
        not bool(((alive == 0) | (alive == 1)).all())
        or not bool(((next_alive == 0) | (next_alive == 1)).all())
        or bool((next_alive > alive).any())
        or not bool((alive[1:] == next_alive[:-1]).all())
        or bool(((alive == 0) & (reward != 0)).any())
    ):
        raise ValueError("terminal recovery must not revive or reward dead worlds")
    result = {name: value[:prefix_frames].clone() for name, value in rollout.items()}
    suffix = torch.zeros_like(reward[0])
    for index in range(len(reward) - 1, prefix_frames - 1, -1):
        suffix = reward[index] + gamma * suffix * next_alive[index]
    result["reward"][-1] += gamma * suffix * next_alive[prefix_frames - 1]
    if not bool(torch.isfinite(result["reward"]).all()):
        raise ValueError("discounted recovery credit overflow")
    result["next_alive"][-1].zero_()
    return result
