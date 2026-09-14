"""Trainable full kick actor with explicit contextual input contract.

This module constructs numerical models only, never a controller or executor.
Reference parameters must already be independently qualified. Importing weights
does not qualify changed precision, contextual inference, learning or retention.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

SCHEMA = "soccer.full_kick_actor_critic.554x29.fp64accum.v1"
LEARNING_FEATURE = 553


def full_kick_observation(
    *,
    reference_observation: NDArray[np.floating],
    goal_relative_position: NDArray[np.floating],
    local_root_velocity: NDArray[np.floating],
    learning_active: bool,
) -> NDArray[np.float32]:
    """Preserve 547 pretrained features; append goal/10, velocity/3, learning bit.

    Goal and root velocity must be expressed in the same current pelvis frame
    as the reference's ball observation. No future ball prediction belongs here.
    The learning bit is excluded from both actor and critic numerical inputs.
    """
    if type(learning_active) is not bool:
        raise ValueError("explicit binary learning activity required")
    for value, size in (
        (reference_observation, 547),
        (goal_relative_position, 3),
        (local_root_velocity, 3),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (size,)
            or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not np.isfinite(value).all()
            or (np.abs(value) > 1e4).any()
        ):
            raise ValueError("bounded finite full-kick observation arrays required")
    return np.concatenate(
        [
            reference_observation.astype(np.float32),
            np.clip(goal_relative_position.astype(np.float32) / 10, -1, 1),
            np.clip(local_root_velocity.astype(np.float32) / 3, -1, 1),
            np.asarray([float(learning_active)], dtype=np.float32),
        ]
    ).astype(np.float32)


def build_full_kick_actor_critic(
    reference_state: Mapping[str, NDArray[np.float32]],
) -> Any:
    """Copy a qualified 547→512→256→128→29 ELU actor for stable accumulation.

    Keys are Sequential ``0/2/4/6.weight/bias`` arrays, not arbitrary pickle or
    an executable model. Six contextual inputs use a separate zero-initialized
    projection, without widening the imported first layer. Parameters, returned
    actions and optimizer states stay float32; actor and critic affine/ELU
    calculations accumulate in float64 to reduce serial-vs-batch likelihood
    drift. This is a distinct precision contract requiring native revalidation,
    not a claim of bit-equivalence to the imported float32 reference.
    All copied actor parameters are trainable; immutable reference artifacts
    remain the caller's protected baseline. Torch is an optional dependency.
    """
    widths = (547, 512, 256, 128, 29)
    shapes = {
        f"{2 * i}.{kind}": shape
        for i in range(4)
        for kind, shape in (
            ("weight", (widths[i + 1], widths[i])),
            ("bias", (widths[i + 1],)),
        )
    }
    if not isinstance(reference_state, Mapping) or set(reference_state) != set(shapes):
        raise ValueError("exact qualified kick reference parameter keys required")
    copied = {}
    for key, shape in shapes.items():
        value = reference_state[key]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != np.float32
            or not np.isfinite(value).all()
            or (np.abs(value) > 1e4).any()
        ):
            raise ValueError("finite float32 reference parameters required")
        copied[key] = value.copy()

    import torch
    from torch import nn

    class FullKickActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            for i in range(4):
                layers.append(
                    nn.Linear(widths[i], widths[i + 1], dtype=torch.float32, device="cpu")
                )
                if i < 3:
                    layers.append(nn.ELU(alpha=1.0))
            self.actor = nn.Sequential(*layers)
            self.actor.load_state_dict({k: torch.from_numpy(v) for k, v in copied.items()})
            self.context_adapter = nn.Linear(6, 512, bias=False, dtype=torch.float32, device="cpu")
            value_head = nn.Linear(128, 1, dtype=torch.float32, device="cpu")
            self.critic = nn.Sequential(
                nn.Linear(553, 256, dtype=torch.float32, device="cpu"),
                nn.ELU(),
                nn.Linear(256, 128, dtype=torch.float32, device="cpu"),
                nn.ELU(),
                value_head,
            )
            self.logstd = nn.Parameter(torch.full((29,), -4.0, dtype=torch.float32, device="cpu"))
            with torch.no_grad():
                self.context_adapter.weight.zero_()
                value_head.weight.zero_()
                value_head.bias.zero_()

        def forward(self, observation: Any) -> tuple[Any, Any]:
            if (
                observation.ndim != 2
                or observation.shape[1] != 554
                or observation.dtype != torch.float32
            ):
                raise ValueError("explicit batch of 554 float32 features required")

            def affine(layer: Any, value: Any) -> Any:
                return nn.functional.linear(
                    value,
                    layer.weight.to(torch.float64),
                    layer.bias.to(torch.float64) if layer.bias is not None else None,
                )

            hidden = affine(self.actor[0], observation[:, :547].to(torch.float64))
            hidden = hidden + affine(
                self.context_adapter, observation[:, 547:553].to(torch.float64)
            )
            for layer in list(self.actor.children())[1:]:
                hidden = affine(layer, hidden) if isinstance(layer, nn.Linear) else layer(hidden)
            value = observation[:, :553].to(torch.float64)
            for layer in self.critic:
                value = affine(layer, value) if isinstance(layer, nn.Linear) else layer(value)
            return hidden.to(torch.float32), value.squeeze(-1).to(torch.float32)

    with torch.random.fork_rng(devices=[]):
        return FullKickActorCritic()
