"""Offline navigation imitation; callers own leases and physical qualification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from rosclaw_soccer.training.local_navigation import build_local_navigation_actor_critic


def fit_navigation_demonstration(
    parent: Mapping[str, np.ndarray],
    observations: np.ndarray,
    velocity_deltas: np.ndarray,
    *,
    steps: int = 256,
    learning_rate: float = 0.001,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit bounded velocity outputs, leaving critic and exploration scale frozen."""
    if (
        type(steps) is not int
        or not 1 <= steps <= 1024
        or type(learning_rate) not in (int, float)
        or not np.isfinite(learning_rate)
        or not 1e-5 <= learning_rate <= 0.003
    ):
        raise ValueError("bounded explicit imitation optimizer required")
    for value, width in ((observations, 39), (velocity_deltas, 3)):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float32
            or value.ndim != 2
            or value.shape[1] != width
            or not 1 <= len(value) <= 4096
            or not np.isfinite(value).all()
            or np.any(np.abs(value) > 10)
        ):
            raise ValueError("finite aligned float32 navigation examples required")
    if (
        len(observations) != len(velocity_deltas)
        or np.any(np.linalg.norm(velocity_deltas[:, :2], axis=1) > 0.250001)
        or np.any(np.abs(velocity_deltas[:, 2]) > 0.400001)
    ):
        raise ValueError("demonstration exceeds navigation authority or is unaligned")
    import torch

    # Initialization is overwritten by the supplied parent. Do not advance the
    # caller's exploration stream merely to allocate this private CPU model.
    with torch.random.fork_rng(devices=[]):
        model = build_local_navigation_actor_critic()
    expected = model.state_dict()
    if (
        not isinstance(parent, Mapping)
        or set(parent) != set(expected)
        or any(
            not isinstance(parent[k], np.ndarray)
            or parent[k].dtype != np.float32
            or parent[k].shape != tuple(v.shape)
            or not np.isfinite(parent[k]).all()
            or np.any(np.abs(parent[k]) > 100)
            for k, v in expected.items()
        )
    ):
        raise ValueError("bounded complete navigation parent state required")
    model.load_state_dict({k: torch.tensor(v.copy()) for k, v in parent.items()})
    model.requires_grad_(False)
    model.actor.requires_grad_(True)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=learning_rate)
    x, target = torch.tensor(observations.copy()), torch.tensor(velocity_deltas.copy())

    def loss_value() -> Any:
        raw = model.actor(x)
        delta = raw.tanh() * torch.tensor((0.25, 0.25, 0.4))
        scale = (0.25 / delta[:, :2].norm(dim=1).clamp_min(1e-8)).clamp(max=1)
        bounded = torch.cat((delta[:, :2] * scale[:, None], delta[:, 2:]), dim=1)
        return (bounded - target).square().mean()

    before = float(loss_value().detach())
    for _ in range(steps):
        loss = loss_value()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite imitation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("nonfinite imitation gradient")
        optimizer.step()
    weights = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
    if any(not np.isfinite(v).all() or np.any(np.abs(v) > 100) for v in weights.values()):
        raise FloatingPointError("unbounded imitation checkpoint")
    assert all(np.array_equal(parent[k], weights[k]) for k in weights if not k.startswith("actor."))
    return weights, dict(
        schema="soccer.navigation_demonstration_imitation.v1",
        optimizer_steps=steps,
        samples=len(observations),
        learning_rate=learning_rate,
        mse_before=before,
        mse_after=float(loss_value().detach()),
        on_policy=False,
        critic_and_exploration_frozen=True,
        activation_ceiling="SIM_ONLY",
        promotion_eligible=False,
    )
