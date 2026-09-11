"""Contact-local hint for a separate rolling redirect task, not stop-control.

Reward measured velocity change toward the intended target during foot contact.
No success, causal attribution, teammate reception or promotion is inferred.
"""

from __future__ import annotations

from typing import Any


def rolling_redirect_contact_reward(
    *, velocity_before: Any, velocity_after: Any, target_delta_xy: Any, foot_contact: Any
) -> Any:
    import torch

    for value, width in ((velocity_before, 3), (velocity_after, 3), (target_delta_xy, 2)):
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.shape[1] != width
            or not 1 <= len(value) <= 4096
            or value.dtype not in (torch.float32, torch.float64)
            or value.layout != torch.strided
            or value.requires_grad
            or not bool(torch.isfinite(value).all())
            or bool((value.abs() > 100).any())
        ):
            raise ValueError("detached bounded redirect vectors required")
    if (
        velocity_after.shape != velocity_before.shape
        or len(target_delta_xy) != len(velocity_before)
        or any(
            x.device != velocity_before.device or x.dtype != velocity_before.dtype
            for x in (velocity_after, target_delta_xy)
        )
        or not isinstance(foot_contact, torch.Tensor)
        or foot_contact.shape != (len(velocity_before),)
        or foot_contact.dtype != torch.bool
        or foot_contact.device != velocity_before.device
        or foot_contact.layout != torch.strided
    ):
        raise ValueError("aligned physical contact and redirect vectors required")
    distance = torch.linalg.vector_norm(target_delta_xy, dim=1)
    if bool((distance < 1e-6).any()):
        raise ValueError("nonzero intended redirect direction required")
    direction = target_delta_xy / distance[:, None]
    toward_change = ((velocity_after[:, :2] - velocity_before[:, :2]) * direction).sum(1)
    upward_change = (velocity_after[:, 2] - velocity_before[:, 2]).clamp(0, 0.2)
    return torch.where(foot_contact, 2 * toward_change.clamp(-0.2, 0.2) - 2 * upward_change, 0.0)
