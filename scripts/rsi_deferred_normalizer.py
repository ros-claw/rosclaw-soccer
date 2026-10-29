"""Fixed-rollout single-rank normalization experiment; simulation only."""

import torch
from rsl_rl.modules.normalization import EmpiricalNormalization


class DeferredNormalization(EmpiricalNormalization):
    """Collect observation batches but keep transform fixed until PPO finishes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending = []

    @torch.no_grad()
    def update(self, x):
        if not self.training or (self.until is not None and self.count >= self.until):
            return
        if x.ndim != 2 or x.shape[0] != 16 or not torch.isfinite(x).all():
            raise ValueError("finite fixed sixteen-world observation batch required")
        self._pending.append(x.detach().clone())
        if len(self._pending) > 24:
            raise ValueError("normalizer was not flushed after a 24-step rollout")

    @torch.no_grad()
    def flush(self):
        if len(self._pending) != 24:
            raise ValueError("complete 24-step rollout required before normalization update")
        values = torch.cat(self._pending, dim=0)
        super().update(values)
        self._pending.clear()


def install_deferred_normalizers(algorithm):
    result = []
    for model in (algorithm.actor, algorithm.critic):
        old = model.obs_normalizer
        if type(old) is not EmpiricalNormalization or not model.obs_normalization:
            raise ValueError("expected enabled unmodified empirical normalizer")
        new = DeferredNormalization(tuple(old._mean.shape[1:]), eps=old.eps, until=old.until)
        new = new.to(device=old._mean.device, dtype=old._mean.dtype)
        new.load_state_dict(old.state_dict(), strict=True)
        new.train(old.training)
        model.obs_normalizer = new
        result.append(new)
    return result
