"""Experimental fixed-batch distributed normalization for the MJWarp smoke.

All ranks must call each normalizer in the same order with equal batch shapes.
This is simulation-only research code, not a generic distributed RL backend.
"""

import torch
import torch.distributed as dist
from rsl_rl.modules.normalization import EmpiricalNormalization


class FourRankNormalization(EmpiricalNormalization):
    """Update each rank using the same rank-ordered global observation batch."""

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        if not self.training or (self.until is not None and self.count >= self.until):
            return
        if not dist.is_initialized() or dist.get_world_size() != 4 or x.shape[0] != 16:
            raise ValueError("four initialized ranks with fixed sixteen-sample batches required")
        gathered = [torch.empty_like(x) for _ in range(4)]
        dist.all_gather(gathered, x.contiguous())
        super().update(torch.cat(gathered, dim=0))


def install_four_rank_normalizers(algorithm) -> None:
    """Replace only the expected actor and critic empirical normalizers."""
    for model in (algorithm.actor, algorithm.critic):
        old = model.obs_normalizer
        if type(old) is not EmpiricalNormalization or not model.obs_normalization:
            raise ValueError("expected enabled, unmodified RSL-RL empirical normalizer")
        new = FourRankNormalization(tuple(old._mean.shape[1:]), eps=old.eps, until=old.until)
        new = new.to(device=old._mean.device, dtype=old._mean.dtype)
        new.load_state_dict(old.state_dict(), strict=True)
        new.train(old.training)
        model.obs_normalizer = new
