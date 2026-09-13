"""Bounded carry navigation residual math, not a movement command or authority."""

from typing import Any


def apply_task_space_navigation(*, raw: Any, nominal: Any, previous: Any, closed: Any) -> Any:
    """Compose learner correction with nominal navigation before existing guards.

    All units follow the caller's qualified world-frame navigation residual:
    XY m/s and yaw rad/s. Absolute component limits(.7,.7,.8), planar change
    norm<=.012 per50Hz tick, yaw change<=.04. Caller separately clips total
    locomotion command and handles pitch/teammate avoidance. Closed/zero
    corrections return nominal exactly; they do not stop the nominal controller.
    """
    import torch

    if (
        not isinstance(raw, torch.Tensor)
        or raw.ndim != 2
        or raw.shape[1] != 3
        or not 1 <= len(raw) <= 4096
        or raw.dtype != torch.float32
    ):
        raise ValueError("explicit three-axis float32 navigation latents required")
    for value in (raw, nominal, previous):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != raw.shape
            or value.dtype != raw.dtype
            or value.device != raw.device
            or value.layout != torch.strided
            or value.requires_grad
            or not bool(torch.isfinite(value).all())
            or bool((value.abs() > 1e6).any())
        ):
            raise ValueError("aligned finite detached navigation values required")
    if (
        not isinstance(closed, torch.Tensor)
        or closed.shape != (len(raw),)
        or closed.device != raw.device
        or closed.dtype != torch.bool
        or closed.layout != torch.strided
    ):
        raise ValueError("aligned navigation influence mask required")
    limits = raw.new_tensor([0.7, 0.7, 0.8])
    nominal_step = nominal - previous
    if (
        bool((nominal.abs() > limits + 1e-6).any())
        or bool((previous.abs() > limits + 1e-6).any())
        or bool((torch.linalg.vector_norm(nominal_step[:, :2], dim=1) > 0.012001).any())
        or bool((nominal_step[:, 2].abs() > 0.040001).any())
    ):
        raise ValueError("nominal navigation violates original absolute/slew contract")
    proposal = torch.clamp(nominal + 0.2 * limits * torch.tanh(raw), -limits, limits)
    step = proposal - previous
    planar_norm = torch.linalg.vector_norm(step[:, :2], dim=1, keepdim=True)
    planar = step[:, :2] * (0.012 / planar_norm.clamp_min(1e-12)).clamp(max=1)
    result = previous + torch.cat((planar, step[:, 2:3].clamp(-0.04, 0.04)), dim=1)
    inactive = closed | (raw == 0).all(1)
    return torch.where(inactive[:, None], nominal, result).clone()
