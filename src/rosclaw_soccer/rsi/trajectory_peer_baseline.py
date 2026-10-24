"""Action-independent peer-return baseline for small whole-trajectory RL batches.

Grouping is by an externally authenticated identical initial condition, never
by outcomes. A trajectory's baseline excludes ALL its own returns. Different
lengths remain ragged; no truncated/failure trajectory is padded or held.
This module does not admit physical data, change rewards, or run an optimizer.
"""

from collections import Counter

import torch


def peer_return_baselines(
    returns: list[torch.Tensor], initial_groups: list[str]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if (
        not 2 <= len(returns) <= 4096
        or len(initial_groups) != len(returns)
        or any(type(g) is not str or not g for g in initial_groups)
        or any(n < 2 for n in Counter(initial_groups).values())
        or any(
            not isinstance(r, torch.Tensor)
            or r.ndim != 1
            or not 1 <= len(r) <= 300
            or r.dtype != torch.float32
            or r.device.type != "cpu"
            or r.requires_grad
            or not torch.isfinite(r).all()
            for r in returns
        )
    ):
        raise ValueError(
            "finite whole observed CPU returns and >=2 peers per bound initial required"
        )
    baselines, advantages = [], []
    for i, own in enumerate(returns):
        summed, counts = torch.zeros_like(own), torch.zeros_like(own)
        for j, peer in enumerate(returns):
            if j == i or initial_groups[j] != initial_groups[i]:
                continue
            n = min(len(own), len(peer))
            summed[:n] += peer[:n]
            counts[:n] += 1
        # No observed peer at a late time => declared zero baseline, not a
        # fictitious peer transition/recovery or an extended reward sequence.
        baseline = summed / counts.clamp_min(1)
        baselines.append(baseline)
        advantages.append(own - baseline)
    return baselines, advantages
