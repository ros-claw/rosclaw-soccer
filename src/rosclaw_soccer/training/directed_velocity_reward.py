"""Bounded contact-only target-velocity shaping, never a success certificate."""

from typing import Any


def contact_velocity_progress(before: Any, after: Any, target: Any, contact: Any) -> Any:
    """Reward reduced planar vector error, including unwanted lateral velocity.

    Inputs are detached, aligned N x 2 float32/64 world-frame velocities and an
    N-element boolean contact mask. A caller must additionally mask unsafe,
    illegal or contaminated episodes. A positive value does not prove that this
    contact caused the measured change or that the ball reaches a teammate.
    """
    import torch

    if any(
        not isinstance(v, torch.Tensor)
        or v.ndim != 2
        or v.shape[1] != 2
        or not 1 <= len(v) <= 4096
        or v.dtype not in (torch.float32, torch.float64)
        or v.requires_grad
        or not bool(torch.isfinite(v).all())
        for v in (before, after, target)
    ):
        raise ValueError("finite detached planar velocities required")
    if any(
        v.shape != before.shape or v.device != before.device or v.dtype != before.dtype
        for v in (after, target)
    ) or (
        not isinstance(contact, torch.Tensor)
        or contact.shape != (len(before),)
        or contact.device != before.device
        or contact.dtype != torch.bool
    ):
        raise ValueError("aligned velocities and boolean contact mask required")
    reduction = torch.linalg.vector_norm(before - target, dim=1) - torch.linalg.vector_norm(
        after - target, dim=1
    )
    return torch.where(contact, reduction.clamp(-0.3, 0.3), 0.0)
