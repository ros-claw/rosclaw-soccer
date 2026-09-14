"""Explicit previous-action-conditioned Gaussian kick proposals, SIM_ONLY.

This changes the policy distribution, not a post-sampling actuator filter.
Previous sampled raw actions are observable policy state; their transition is
the sampled action itself, independent of the next model's parameters. No
unrecorded noise process or hidden recurrent cache is introduced.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .full_kick_learning import build_full_kick_actor_critic

SCHEMA = "soccer.autoregressive_kick_actor_critic.583x29.fp64accum.v1"


def autoregressive_kick_observation(
    *, base_observation: NDArray[np.float32], previous_raw_action: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Append the actual preceding sampled raw action, not a shadow prediction.

    At entry the caller must bind an explicit, parameter-independent reset
    value (for example the frozen teacher proposal); inactive phases use zero.
    Raw policy action and bounded motor target are different quantities. The
    original 547-feature body history must still contain actual applied targets.
    """
    for value, size in ((base_observation, 554), (previous_raw_action, 29)):
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (size,)
            or value.dtype != np.float32
            or not np.isfinite(value).all()
            or (np.abs(value) > 1e4).any()
        ):
            raise ValueError("bounded finite float32 autoregressive observations required")
    if base_observation[553] not in (0, 1):
        raise ValueError("binary action-ownership feature required")
    if base_observation[553] == 0 and np.any(previous_raw_action != 0):
        raise ValueError("inactive autoregressive history must be reset")
    return np.concatenate((base_observation, previous_raw_action)).astype(np.float32)


def build_autoregressive_kick_actor_critic(
    reference_state: Mapping[str, NDArray[np.float32]], *, continuation_factor: float = 0.25
) -> Any:
    """Conditional mean = (1-rho)*network + rho*previous raw action when active.

    All network parameters remain trainable; rho is a bounded serialized
    configuration buffer. ``logstd`` is the conditional innovation scale,
    not a claimed stationary marginal standard deviation. rho=0 is the exact
    numerical base policy. Nonzero rho changes deterministic tracking as well
    as exploration and therefore requires new native qualification. The critic
    retains the original body/goal baseline and does not consume the appended
    raw-action feature. This is not an implementation of FFT pink-noise PPO.
    """
    if (
        type(continuation_factor) not in (int, float)
        or not math.isfinite(continuation_factor)
        or not 0 <= continuation_factor <= 0.9
    ):
        raise ValueError("continuation factor must be finite in [0, 0.9]")
    import torch
    from torch import nn

    base = build_full_kick_actor_critic(reference_state)

    class AutoregressiveKickActorCritic(nn.Module):
        continuation_factor: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.base = base
            self.register_buffer(
                "continuation_factor",
                torch.tensor(continuation_factor, dtype=torch.float64, device="cpu"),
            )

        @property
        def logstd(self) -> Any:
            return self.base.logstd

        def forward(self, observation: Any) -> tuple[Any, Any]:
            if (
                observation.ndim != 2
                or observation.shape[1] != 583
                or observation.dtype != torch.float32
                or not bool(torch.isfinite(observation).all())
                or bool((observation.abs() > 1e4).any())
                or not bool(((observation[:, 553] == 0) | (observation[:, 553] == 1)).all())
                or not bool(torch.isfinite(self.continuation_factor))
                or not 0 <= float(self.continuation_factor) <= 0.9
            ):
                raise ValueError("validated 583-feature conditional action input required")
            active = observation[:, 553:554] == 1
            previous = observation[:, 554:]
            if bool((previous[~active[:, 0]] != 0).any()):
                raise ValueError("inactive autoregressive history must be reset")
            mean, value = self.base(observation[:, :554])
            rho = active.to(torch.float64) * self.continuation_factor
            conditional_mean = (1 - rho) * mean.to(torch.float64) + rho * previous.to(torch.float64)
            return conditional_mean.to(torch.float32), value

    return AutoregressiveKickActorCritic()
