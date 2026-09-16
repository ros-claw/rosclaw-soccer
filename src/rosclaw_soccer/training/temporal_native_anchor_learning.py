"""Explicit residual-history Gaussian proposals around current native actions.

Only numerical policy math lives here. The caller owns the frozen ONNX inference,
actual sampled-action history, episode resets, optimizer and execution boundary.
No hidden recurrence, simulator state mutation or hardware control is provided.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .native_anchored_kick_learning import build_native_anchored_kick_actor_critic

SCHEMA = "soccer.temporal_native_anchor_actor_critic.612x29.v1"
LEARNING_FEATURE = 582


def temporal_native_anchor_observation(
    *, base_observation: NDArray[np.float32], previous_residual: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Append actual previous raw action minus its same-frame native action.

    The 583-feature prefix uses native-anchor layout, not autoregressive583.
    History starts at zero and resets outside the caller's declared learning
    segment. It must not be reconstructed using a subsequently updated model.
    These input conventions do not grant execution or policy admission.
    """
    for value, size in ((base_observation, 583), (previous_residual, 29)):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float32
            or value.shape != (size,)
            or not np.isfinite(value).all()
            or (np.abs(value) > 1e4).any()
        ):
            raise ValueError("bounded finite float32 native context and residual history required")
    if (
        (np.abs(base_observation[547:553]) > 1).any()
        or base_observation[582] not in (0, 1)
        or (base_observation[582] == 0 and np.any(previous_residual != 0))
    ):
        raise ValueError("scaled context, binary segment bit and inactive history reset required")
    return np.concatenate((base_observation, previous_residual))


def build_temporal_native_anchor_actor_critic(*, continuation_factor: float = 0.75) -> Any:
    """Condition on prior feedback without filtering the native motion phase.

    Inside the learning segment:
      mean = native_now + (1-rho)*(base_mean-native_now) + rho*previous_residual.
    Outside it, history must be zero and the original base mean is returned.
    Thus the segment bit changes the conditional distribution, unlike the base
    model's metadata-only bit; it still cannot authorize any motor execution.

    rho is a serialized non-trainable buffer. Zero rho exactly preserves the
    base actor; zero feedback/history preserves native actions even at nonzero
    rho. The critic retains the base context and does not read residual history.
    logstd denotes conditional innovation noise, not marginal residual spread.
    The history term and Gaussian samples are not bounded by the base model's
    0.1 mean-feedback bound. Caller-owned clipping and physical gates still apply.
    Torch is optional until construction.
    """
    if (
        type(continuation_factor) not in (int, float)
        or not math.isfinite(continuation_factor)
        or not 0 <= continuation_factor <= 0.9
    ):
        raise ValueError("finite residual continuation in [0, 0.9] required")
    import torch
    from torch import nn

    class TemporalNativeAnchorActorCritic(nn.Module):
        rho: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.base = build_native_anchored_kick_actor_critic()
            self.register_buffer("rho", torch.tensor(continuation_factor, dtype=torch.float32))

        @property
        def logstd(self) -> Any:
            return self.base.logstd

        def forward(self, observation: Any) -> tuple[Any, Any]:
            if (
                not isinstance(observation, torch.Tensor)
                or observation.ndim != 2
                or observation.shape[1] != 612
                or observation.dtype != torch.float32
                or not bool(torch.isfinite(observation).all())
                or bool((observation.abs() > 1e4).any())
                or not bool(torch.isfinite(self.rho))
                or not 0 <= float(self.rho) <= 0.9
            ):
                raise ValueError("finite bounded native-anchor612 input and continuation required")
            active = observation[:, 582:583] == 1
            previous = observation[:, 583:]
            if bool((previous[~active[:, 0]] != 0).any()):
                raise ValueError("inactive residual history must be reset")
            mean, value = self.base(observation[:, :583])
            if float(self.rho) == 0:
                return mean, value
            native = observation[:, 553:582]
            feedback = mean - native
            continued = native + (1 - self.rho) * feedback + self.rho * previous
            return torch.where(active, continued, mean), value

    return TemporalNativeAnchorActorCritic()
