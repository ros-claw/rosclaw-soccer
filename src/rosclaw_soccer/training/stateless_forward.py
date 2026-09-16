"""Opt-in singleton arithmetic for declared stateless actor-critic models.

Small exploration variances can amplify float32 GEMM differences between a
single rollout row and a learner batch. This preserves the rollout's row-wise
arithmetic and autograd; it does not relax likelihood checks or change weights.
Callers must establish that custom modules have no cross-row or hidden state.
This is not an adapter for a live recurrent locomotion controller.
"""

from __future__ import annotations

from typing import Any


def stateless_actor_critic_rows(agent: Any, observations: Any) -> tuple[Any, Any]:
    """Evaluate independent float32 rows; explicitly reject known coupled layers."""
    import torch

    if (
        not isinstance(agent, torch.nn.Module)
        or not isinstance(observations, torch.Tensor)
        or observations.ndim != 2
        or not 1 <= observations.shape[0] <= 65536
        or observations.shape[1] < 1
        or observations.dtype != torch.float32
        or not bool(torch.isfinite(observations).all())
    ):
        raise ValueError("bounded finite float32 rows and a declared stateless module required")
    batch_norm = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
    )
    dropout = (
        torch.nn.Dropout,
        torch.nn.Dropout1d,
        torch.nn.Dropout2d,
        torch.nn.Dropout3d,
        torch.nn.AlphaDropout,
        torch.nn.FeatureAlphaDropout,
    )
    for module in agent.modules():
        if (
            isinstance(module, (torch.nn.RNNBase, torch.nn.MultiheadAttention))
            or isinstance(module, batch_norm)
            and (module.training or not module.track_running_stats)
            or isinstance(module, dropout)
            and module.training
            and module.p > 0
        ):
            raise ValueError("row-wise PPO requires stateless, non-stochastic feed-forward layers")
    means, values = [], []
    action_size = None
    for row in observations:
        result = agent(row.unsqueeze(0))
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("actor-critic must return a mean/value tuple")
        mean, value = result
        if (
            not isinstance(mean, torch.Tensor)
            or not isinstance(value, torch.Tensor)
            or mean.ndim != 2
            or mean.shape[0] != 1
            or not 1 <= mean.shape[1] <= 1024
            or value.shape != (1,)
            or mean.dtype != torch.float32
            or value.dtype != torch.float32
            or mean.device != observations.device
            or value.device != observations.device
            or action_size is not None
            and mean.shape[1] != action_size
            or not bool(torch.isfinite(mean).all())
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("finite same-device singleton actor-critic outputs required")
        action_size = mean.shape[1]
        means.append(mean)
        values.append(value)
    return torch.cat(means, dim=0), torch.cat(values, dim=0)
