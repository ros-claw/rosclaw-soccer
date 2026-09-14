"""Reference-centered conditional exploration; numeric SIM_ONLY proposals.

Unlike smoothing the entire raw action, this only continues the previous
deviation from an immutable reference. It preserves that reference's fast mean
trajectory at initialization when no stochastic actions are sampled. History
must be reconstructed against a FROZEN reference, never the changing learner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .autoregressive_kick_learning import autoregressive_kick_observation
from .full_kick_learning import build_full_kick_actor_critic

SCHEMA = "soccer.reference_relative_kick_actor_critic.583x29.fp64accum.v1"


def reference_relative_kick_observation(
    *,
    base_observation: NDArray[np.float32],
    previous_raw_action: NDArray[np.float32],
    previous_frozen_reference_mean: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Append raw-minus-frozen-reference history with an explicit entry reset.

    The caller records both preceding quantities and binds reference weights
    and inference inputs. At entry and when inactive, BOTH histories are zero.
    They are not motor targets, current learner means, or post-filtered actions.
    This function checks numbers, not the caller's provenance claims.
    """
    for value in (previous_raw_action, previous_frozen_reference_mean):
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (29,)
            or value.dtype != np.float32
            or not np.isfinite(value).all()
            or (np.abs(value) > 5e3).any()
        ):
            raise ValueError("finite bounded float32 preceding action and reference required")
    # Reuse base/flag validation, without mislabelling the reference as an action.
    checked = autoregressive_kick_observation(
        base_observation=base_observation, previous_raw_action=np.zeros(29, dtype=np.float32)
    )
    if checked[553] == 0 and (
        np.any(previous_raw_action != 0) or np.any(previous_frozen_reference_mean != 0)
    ):
        raise ValueError("inactive raw and frozen reference histories must both be reset")
    delta = (
        previous_raw_action.astype(np.float64) - previous_frozen_reference_mean.astype(np.float64)
    ).astype(np.float32)
    checked[554:] = delta
    return checked


def build_reference_relative_kick_actor_critic(
    reference_state: Mapping[str, NDArray[np.float32]], *, continuation_factor: float = 0.25
) -> Any:
    """Mean = current network mean + rho * previous reference-relative action.

    The separate frozen reference and its history transition belong to the
    collector; they MUST NOT change with optimizer parameters. There is no
    hidden recurrent cache. The critic retains the body/goal-only baseline.
    logstd denotes conditional innovation scale, not stationary marginal noise.
    Learned means may deviate from the frozen reference and accumulate history;
    initialization equivalence does not guarantee trained stability or success.
    This is a new policy schema, NOT interchangeable with absolute-action AR.
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

    class ReferenceRelativeKickActorCritic(nn.Module):
        continuation_factor: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.base = base
            self.register_buffer(
                "continuation_factor", torch.tensor(continuation_factor, dtype=torch.float64)
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
                or bool((observation[observation[:, 553] == 0, 554:] != 0).any())
            ):
                raise ValueError("validated reference-relative action history required")
            mean, value = self.base(observation[:, :554])
            correction = self.continuation_factor * observation[:, 554:].to(torch.float64)
            return (mean.to(torch.float64) + correction).to(torch.float32), value

    return ReferenceRelativeKickActorCritic()
