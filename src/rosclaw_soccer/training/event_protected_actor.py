"""Protect a frozen action prefix while learning an event-conditioned successor.

This is optimizer infrastructure, not a contact detector or activation gate.
The collector must bind the final binary observation feature to real, causal
event evidence. It must separately validate successor dynamics and transitions.
No robot identity, world step, hardware command or policy promotion lives here.
"""

from __future__ import annotations

import copy
from typing import Any


def build_event_protected_actor_critic(
    seed: Any, *, observation_size: int, action_size: int
) -> Any:
    """Clone an actor/critic; binary final input selects a plastic action head.

    Before the event, deterministic action means come from the immutable clone.
    The critic can learn throughout the episode. Exploration standard deviation
    is frozen globally, so PPO cannot modify the protected action distribution
    indirectly through a shared logstd. Stochastic samples are NOT claimed to
    equal deterministic actions, and post-event safety/retention is unproven.

    ``observation_size`` excludes the additional binary event feature. The seed
    must provide ``forward -> (mean, value)`` and a vector ``logstd``. Its state
    must be finite CPU float32; the returned model can subsequently be moved.
    Torch remains an optional lazy dependency.
    """
    import torch

    if (
        type(observation_size) is not int
        or type(action_size) is not int
        or not 1 <= observation_size <= 1024
        or not 1 <= action_size <= 256
        or not isinstance(seed, torch.nn.Module)
        or not isinstance(getattr(seed, "logstd", None), torch.Tensor)
        or seed.logstd.shape != (action_size,)
        or not seed.state_dict()
        or any(
            value.dtype != torch.float32
            or value.device.type != "cpu"
            or value.layout != torch.strided
            or not bool(torch.isfinite(value).all())
            for value in seed.state_dict().values()
        )
    ):
        raise ValueError("bounded dimensions and finite CPU float32 actor/critic seed required")

    class EventProtectedActorCritic(torch.nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.anchor = copy.deepcopy(seed).eval().requires_grad_(False)
            self.plastic = copy.deepcopy(seed).eval().requires_grad_(True)
            self.plastic.logstd.requires_grad_(False)
            self.logstd = torch.nn.Parameter(seed.logstd.detach().clone(), requires_grad=False)

        def train(self, mode: bool = True) -> Any:
            super().train(mode)
            # A frozen prefix must not acquire training-mode dropout or mutable
            # normalization behavior when the parent learner enters train mode.
            self.anchor.eval()
            return self

        def forward(self, observation: Any) -> tuple[Any, Any]:
            if (
                not isinstance(observation, torch.Tensor)
                or observation.ndim != 2
                or observation.shape[1] != observation_size + 1
                or not 1 <= observation.shape[0] <= 65536
                or observation.dtype != self.logstd.dtype
                or observation.device != self.logstd.device
                or observation.layout != torch.strided
                or not bool(torch.isfinite(observation).all())
                or bool((observation.abs() > 10).any())
                or not bool(((observation[:, -1] == 0) | (observation[:, -1] == 1)).all())
            ):
                raise ValueError("bounded observations with an exact binary event feature required")
            body = observation[:, :-1].contiguous()
            with torch.no_grad():
                protected_mean, _ = self.anchor(body)
            plastic_mean, value = self.plastic(body)
            if (
                protected_mean.shape != (len(body), action_size)
                or plastic_mean.shape != protected_mean.shape
                or value.shape != (len(body),)
                or not bool(
                    torch.isfinite(protected_mean).all()
                    and torch.isfinite(plastic_mean).all()
                    and torch.isfinite(value).all()
                )
            ):
                raise FloatingPointError("invalid protected or plastic actor/critic output")
            return torch.where(observation[:, -1:] == 1, plastic_mean, protected_mean), value

    return EventProtectedActorCritic()
