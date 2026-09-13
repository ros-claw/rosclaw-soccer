"""Detached group-relative outcome signals for simulation research.

Grouping is explicit training metadata, never inferred from observation shape.
The caller owns the assertion that group members represent comparable tasks.
No rewards, samples, policies, physics state or promotion status are changed.
"""

from typing import Any


def group_relative_advantages(returns: Any, group_ids: Any) -> Any:
    """Subtract each group's mean and divide by its population standard deviation.

    Constant-outcome groups produce zero. All members, including failures, stay
    in the group. Sorting and per-segment reductions avoid floating-point atomic
    scatter sums; this is not a cross-backend bitwise reproducibility promise.
    The operation is an outcome estimator, not an implementation of full GRPO.
    """
    import torch

    if (
        not isinstance(returns, torch.Tensor)
        or returns.ndim != 1
        or not 2 <= len(returns) <= 4096
        or returns.dtype not in (torch.float32, torch.float64)
        or returns.layout != torch.strided
        or returns.requires_grad
        or not bool(torch.isfinite(returns).all())
        or bool((returns.abs() > 1e6).any())
        or not isinstance(group_ids, torch.Tensor)
        or group_ids.shape != returns.shape
        or group_ids.dtype != torch.int64
        or group_ids.layout != torch.strided
        or group_ids.device != returns.device
        or bool((group_ids < 0).any())
    ):
        raise ValueError("bounded detached returns and aligned int64 task groups required")
    order = torch.argsort(group_ids, stable=True)
    _, counts = torch.unique_consecutive(group_ids[order], return_counts=True)
    if bool((counts < 2).any()):
        raise ValueError("each task group requires at least two attempts")
    sorted_returns: Any = returns[order]
    pieces = sorted_returns.split(counts.cpu().tolist())
    normalized_pieces = []
    for piece in pieces:
        # Remove the common offset before reduction. In particular, a constant
        # group must remain exactly zero even if a large raw mean would round.
        shifted = piece - piece[0]
        normalized_pieces.append(
            (shifted - shifted.mean()) / shifted.std(correction=0).clamp_min(1e-6)
        )
    normalized = torch.cat(normalized_pieces)
    if not bool(torch.isfinite(normalized).all()):
        raise FloatingPointError("nonfinite group-relative outcome signal")
    result = torch.empty_like(returns)
    result[order] = normalized
    return result
