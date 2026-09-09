"""Stop-control curriculum rewards; not a pass/shot or promotion evaluator.

Touching and then losing the ball must not be scored worse than never trying
to touch it. Full credit still requires a clean, safe, observed control window.
Callers derive every mask from physical contacts, not tactical intentions.
"""

from __future__ import annotations

import math
from typing import Any


def controlled_reception_mask(
    *,
    seen_foot_contact: Any,
    forbidden_contact: Any,
    body_safe: Any,
    stable_tail: Any,
    first_contact_sec: Any,
    elapsed_sec: float,
    minimum_observation_sec: float = 0.5,
) -> Any:
    import torch

    masks = (seen_foot_contact, forbidden_contact, body_safe, stable_tail)
    if (
        not isinstance(first_contact_sec, torch.Tensor)
        or first_contact_sec.ndim != 1
        or not 1 <= len(first_contact_sec) <= 4096
        or not torch.is_floating_point(first_contact_sec)
        or not bool(torch.isfinite(first_contact_sec).all())
        or any(
            not isinstance(m, torch.Tensor)
            or m.dtype != torch.bool
            or m.shape != first_contact_sec.shape
            or m.device != first_contact_sec.device
            for m in masks
        )
        or type(elapsed_sec) not in (int, float)
        or not math.isfinite(elapsed_sec)
        or not 0 < elapsed_sec <= 60
        or type(minimum_observation_sec) not in (int, float)
        or not math.isfinite(minimum_observation_sec)
        or not 0.3 <= minimum_observation_sec <= 1.0
    ):
        raise ValueError("finite aligned physical reception outcomes required")
    if bool(
        (seen_foot_contact != (first_contact_sec >= 0)).any()
        or ((~seen_foot_contact) & (first_contact_sec != -1)).any()
        or (first_contact_sec > elapsed_sec + 1e-6).any()
    ):
        raise ValueError("reception contact clock disagrees with measured event")
    return (
        seen_foot_contact
        & ~forbidden_contact
        & body_safe
        & stable_tail
        & (elapsed_sec - first_contact_sec >= minimum_observation_sec - 1e-6)
    )


def reception_terminal_reward(**physical_outcomes: Any) -> Any:
    """Same control mask as evaluation; no reward for a truncated late touch."""
    import torch

    controlled = controlled_reception_mask(**physical_outcomes)
    seen = physical_outcomes["seen_foot_contact"]
    return torch.where(controlled, 10.0, torch.where(seen, -1.0, -2.0))


def rolling_reception_contact_reward(
    *, velocity_before: Any, velocity_after: Any, foot_contact: Any
) -> Any:
    """Shape a rolling-ball trap: brake without popping the ball into the body.

    Velocities are measured in the fixed world frame, Z up, at adjacent
    physics steps. This is a training hint during contact, not proof that the
    foot caused every velocity change, and never a successful-reception gate.
    Aerial receptions and deliberate lofted passes require different tasks.
    """
    import torch

    velocities = (velocity_before, velocity_after)
    if (
        any(
            not isinstance(v, torch.Tensor)
            or v.ndim != 2
            or v.shape[1] != 3
            or not 1 <= len(v) <= 4096
            or v.dtype not in (torch.float32, torch.float64)
            or v.requires_grad
            or not bool(torch.isfinite(v).all())
            or bool((v.abs() > 100).any())
            for v in velocities
        )
        or velocity_before.shape != velocity_after.shape
        or velocity_before.device != velocity_after.device
        or velocity_before.dtype != velocity_after.dtype
        or not isinstance(foot_contact, torch.Tensor)
        or foot_contact.dtype != torch.bool
        or foot_contact.shape != (len(velocity_before),)
        or foot_contact.device != velocity_before.device
    ):
        raise ValueError("detached aligned physical ball velocities and foot mask required")
    braking = (
        torch.linalg.vector_norm(velocity_before, dim=1)
        - torch.linalg.vector_norm(velocity_after, dim=1)
    ).clamp(-0.2, 0.2)
    upward_change = (velocity_after[:, 2] - velocity_before[:, 2]).clamp(0, 0.2)
    return torch.where(foot_contact, 2 * braking - 2 * upward_change, 0.0)
