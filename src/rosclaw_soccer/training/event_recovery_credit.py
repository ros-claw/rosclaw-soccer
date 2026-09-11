"""Credit observed recovery to each world's last executed policy action.

The collector, not this arithmetic helper, must prove that later controls do
not execute new policy samples. An event is not an execution authorization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rosclaw_soccer.training.fixed_recovery_credit import fold_fixed_recovery_tail


def fold_event_recovery_tail(
    rollout: Mapping[str, Any], *, prefix_frames: Any, gamma: float
) -> dict[str, Any]:
    """Return padded on-policy prefixes with terminal discounted recovery credit.

    ``prefix_frames`` is one detached int64 count per world, including the last
    executed action. Padding is dead/zero, never a fabricated action or reset.
    """
    import torch

    reward = rollout.get("reward")
    if (
        not isinstance(reward, torch.Tensor)
        or reward.ndim != 2
        or not isinstance(prefix_frames, torch.Tensor)
        or prefix_frames.dtype != torch.int64
        or prefix_frames.layout != torch.strided
        or prefix_frames.shape != (reward.shape[1],)
        or prefix_frames.device != reward.device
        or prefix_frames.requires_grad
        or not 1 <= reward.shape[0] <= 4096
        or not 1 <= reward.shape[1] <= 4096
        or bool((prefix_frames < 1).any())
        or bool((prefix_frames > reward.shape[0]).any())
    ):
        raise ValueError("one bounded executed-prefix count per world required")
    # Reuse all full-rollout, discount, no-revival and dead-reward validation.
    maximum = int(prefix_frames.max())
    result = fold_fixed_recovery_tail(rollout, prefix_frames=maximum, gamma=gamma)
    result["reward"] = reward[:maximum].clone()
    # Vectorize worlds: at most horizon recurrences, not horizon * worlds
    # separate Python/device operations. Preserve the fixed helper's masks.
    credit = torch.zeros_like(reward[0])
    for frame in range(reward.shape[0] - 1, -1, -1):
        credit = torch.where(
            frame >= prefix_frames,
            reward[frame] + gamma * credit * rollout["next_alive"][frame],
            credit,
        )
    worlds = torch.arange(reward.shape[1], device=reward.device)
    last = prefix_frames - 1
    result["reward"][last, worlds] += gamma * credit * rollout["next_alive"][last, worlds]
    result["next_alive"][last, worlds] = 0
    padding = torch.arange(maximum, device=reward.device)[:, None] >= prefix_frames[None]
    for value in result.values():
        value.masked_fill_(padding[:, :, None] if value.ndim == 3 else padding, 0)
    if not bool(torch.isfinite(result["reward"]).all()):
        raise ValueError("event recovery credit overflow")
    return result
