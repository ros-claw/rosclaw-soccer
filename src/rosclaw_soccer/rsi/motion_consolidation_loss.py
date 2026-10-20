"""Masked per-sequence imitation objective; no physics or activation authority."""

from typing import Any


def weighted_motion_token_loss(tokens: Any, targets: Any, mask: Any, weights: Any) -> Any:
    import torch

    if (
        tokens.ndim != 3
        or tokens.shape[0] < 1
        or tokens.shape[1:] != (300, 64)
        or targets.shape != tokens.shape
        or tokens.dtype != torch.float32
        or targets.dtype != torch.float32
        or mask.shape != tokens.shape[:2]
        or mask.dtype != torch.bool
        or weights.shape != (tokens.shape[0],)
        or weights.dtype != torch.float32
        or mask[:, :20].any()
        or not mask.any()
        or not torch.isfinite(tokens).all()
        or not torch.isfinite(targets).all()
        or not torch.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("complete finite masked motion memory batch required")
    per_frame = (tokens - targets).square().mean(-1)
    sequence = (per_frame * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    return (sequence * weights).mean()
