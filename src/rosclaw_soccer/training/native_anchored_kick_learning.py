"""Numerical kick adapter anchored to an independently executed frozen policy.

The caller supplies the actual current frozen action, not a reimplementation of
the native network. No controller, simulator, transport or optimizer is owned.
Zero initialization preserves that supplied action; learned safety, observation
provenance, physical equivalence and retention require independent evidence.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

SCHEMA = "soccer.native_anchored_kick_actor_critic.583x29.v1"
LEARNING_FEATURE = 582
MAXIMUM_MEAN_RESIDUAL = 0.1


def native_anchored_kick_observation(
    *, contextual_observation: NDArray[np.float32], reference_action: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Pack current full554 context + current raw reference action, without history.

    Layout: kick547, goal3, rootvelocity3, current reference action29, learning
    bit. This is NOT the historical autoregressive583 observation contract.
    The reference must come from this player's same-frame frozen native policy.
    Arrays/hashes alone cannot establish that fact or authorize a motor action.
    """
    for value, size in ((contextual_observation, 554), (reference_action, 29)):
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (size,)
            or value.dtype != np.float32
            or not np.isfinite(value).all()
            or (np.abs(value) > 1e4).any()
        ):
            raise ValueError("bounded finite float32 context and current reference required")
    if (np.abs(contextual_observation[547:553]) > 1).any() or contextual_observation[553] not in (
        0,
        1,
    ):
        raise ValueError("scaled current context and explicit binary learning bit required")
    return np.concatenate(
        (contextual_observation[:553], reference_action, contextual_observation[553:])
    ).astype(np.float32)


def build_native_anchored_kick_actor_critic() -> Any:
    """Build a zero-mean-residual adapter; Torch remains optional until called.

    Mean = supplied native raw action + 0.1*tanh(582→128→29 feedback).
    Gaussian samples need not satisfy the mean residual bound. The native
    action mapping, clipping, joint/torque guards and control ownership remain
    caller-owned. The training bit changes neither actor nor critic outputs;
    it is not an execution/admission switch. There is no hidden recurrent state.
    logstd is trainable by default: callers must either include it in the
    optimizer or explicitly freeze it before requesting a PPO update.
    """
    import torch
    from torch import nn

    class NativeAnchoredKickActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.actor = nn.Sequential(nn.Linear(582, 128), nn.Tanh(), nn.Linear(128, 29))
            self.critic = nn.Sequential(nn.Linear(582, 128), nn.Tanh(), nn.Linear(128, 1))
            self.logstd = nn.Parameter(torch.full((29,), -6.0))
            actor_head = cast(nn.Linear, self.actor[-1])
            critic_head = cast(nn.Linear, self.critic[-1])
            nn.init.zeros_(actor_head.weight)
            nn.init.zeros_(actor_head.bias)
            nn.init.zeros_(critic_head.weight)
            nn.init.zeros_(critic_head.bias)

        def forward(self, observation: Any) -> tuple[Any, Any]:
            if (
                not isinstance(observation, torch.Tensor)
                or observation.ndim != 2
                or observation.shape[1] != 583
                or observation.dtype != torch.float32
                or not bool(torch.isfinite(observation).all())
                or bool((observation.abs() > 1e4).any())
                or bool((observation[:, 547:553].abs() > 1).any())
                or not bool(((observation[:, 582] == 0) | (observation[:, 582] == 1)).all())
            ):
                raise ValueError("explicit finite native-anchor583 observation required")
            features = observation[:, :582]
            mean = observation[:, 553:582] + MAXIMUM_MEAN_RESIDUAL * torch.tanh(
                self.actor(features)
            )
            return mean, self.critic(features).squeeze(-1)

    return NativeAnchoredKickActorCritic()
