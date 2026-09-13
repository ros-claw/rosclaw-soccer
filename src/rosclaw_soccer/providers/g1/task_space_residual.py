"""Bounded leg-only force-to-target math; no simulator or actuator ownership."""

from typing import Any


def apply_task_space_residual(
    *,
    raw: Any,
    jacobian: Any,
    foot: Any,
    launch_direction: Any,
    kp: Any,
    nominal: Any,
    previous: Any,
    closed: Any,
) -> tuple[Any, Any]:
    """Map a 3D launch-frame latent through a qualified selected-leg Jacobian.

    Foot IDs are 0/1 for the explicit G1 hg left/right six-joint legs. Callers
    must qualify ancestry and geometry timing. Force norm is at most 60 N;
    total residual remains within .25 rad and .025 rad per control tick.
    This returns target math, never direct force on a ball or floating base.
    Existing final joint/torque guards are still mandatory at execution.
    """
    import torch

    if not isinstance(raw, torch.Tensor) or raw.ndim != 2:
        raise ValueError("batched three-axis latent required")
    n = len(raw)
    if not 1 <= n <= 4096 or raw.dtype != torch.float32:
        raise ValueError("bounded float32 task-space batch required")
    for value, shape in (
        (raw, (n, 3)),
        (jacobian, (n, 3, 6)),
        (launch_direction, (n, 2)),
        (kp, (n, 29)),
        (nominal, (n, 29)),
        (previous, (n, 29)),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != shape
            or value.device != raw.device
            or value.dtype != torch.float32
            or value.requires_grad
            or value.layout != torch.strided
            or not bool(torch.isfinite(value).all())
            or bool((value.abs() > 1e6).any())
        ):
            raise ValueError("finite detached aligned task-space values required")
    for value, dtype in ((foot, torch.int64), (closed, torch.bool)):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (n,)
            or value.device != raw.device
            or value.dtype != dtype
            or value.layout != torch.strided
        ):
            raise ValueError("explicit selected foot and physical closure required")
    if (
        bool(((foot < 0) | (foot > 1)).any())
        or bool(((kp < 1) | (kp > 1000)).any())
        or bool((jacobian.abs() > 2).any())
        or bool((torch.linalg.vector_norm(launch_direction, dim=1) - 1).abs().gt(1e-4).any())
        or bool((nominal.abs() > 0.250001).any())
        or bool((previous.abs() > 0.250001).any())
        or bool(((nominal - previous).abs() > 0.025001).any())
    ):
        raise ValueError("qualified leg mapping and original residual envelope required")
    force = raw.tanh() * 60
    force = force / (torch.linalg.vector_norm(force, dim=1, keepdim=True) / 60).clamp_min(1)
    x, y = launch_direction.unbind(1)
    world_force = torch.stack(
        (x * force[:, 0] - y * force[:, 1], y * force[:, 0] + x * force[:, 1], force[:, 2]),
        dim=1,
    )
    world_force = world_force / (
        torch.linalg.vector_norm(world_force, dim=1, keepdim=True) / 60
    ).clamp_min(1)
    world_force = torch.where(closed[:, None], torch.zeros_like(world_force), world_force)
    indices = foot[:, None] * 6 + torch.arange(6, device=raw.device)[None]
    torque = torch.bmm(jacobian.transpose(1, 2), world_force[:, :, None]).squeeze(-1)
    correction = torch.zeros_like(nominal).scatter(1, indices, torque / kp.gather(1, indices))
    proposal = (nominal + 0.2 * correction).clamp(-0.25, 0.25)
    result = previous + (proposal - previous).clamp(-0.025, 0.025)
    result = torch.where(correction == 0, nominal, result)
    # Disabled/closed teacher must preserve the nominal floating-point path exactly.
    result = torch.where((world_force == 0).all(1)[:, None], nominal, result)
    return result.clone(), world_force.clone()
