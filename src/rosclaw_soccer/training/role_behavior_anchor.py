"""Immutable rehearsal targets for private role PPO; no policy activation.

The anchor retains a reference action distribution on observed old-skill
states. KL regularization is not a guarantee of physical skill retention:
callers must still replay the actual old tasks and reject regressions.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


@dataclass(frozen=True)
class RoleBehaviorAnchor:
    policy: NearBallResidualPolicy
    observations: np.ndarray
    active: np.ndarray
    source_hash: str
    coefficient: float = 100.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.policy, NearBallResidualPolicy)
            or not isinstance(self.source_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.source_hash) is None
            or type(self.coefficient) not in (int, float)
            or not math.isfinite(self.coefficient)
            or not 0.01 <= self.coefficient <= 1000.0
        ):
            raise ValueError("content-bound bounded behavior anchor required")
        obs, active = np.asarray(self.observations), np.asarray(self.active)
        if (
            obs.ndim != 3
            or obs.shape[1:] != (8, 56)
            or not 1 <= len(obs) <= 8192
            or obs.dtype.kind not in "fiu"
            or not np.all(np.isfinite(obs))
            or np.max(np.abs(obs.astype(np.float64))) > 1000
            or active.shape != obs.shape[:2]
            or active.dtype != np.bool_
            or not np.any(active)
        ):
            raise ValueError("finite bounded role observations and explicit active mask required")
        object.__setattr__(
            self,
            "observations",
            np.frombuffer(np.asarray(obs, dtype=np.float64).tobytes(), dtype=np.float64).reshape(
                obs.shape
            ),
        )
        object.__setattr__(
            self, "active", np.frombuffer(active.tobytes(), dtype=np.bool_).reshape(active.shape)
        )

    @property
    def anchor_hash(self) -> str:
        return str(
            hash_json(
                {
                    "schema": "rosclaw_soccer.role_behavior_anchor.v1",
                    "policy": self.policy.policy_hash,
                    "source": self.source_hash,
                    "observations": hash_bytes(self.observations.tobytes()),
                    "shape": self.observations.shape,
                    "active": hash_bytes(self.active.tobytes()),
                    "coefficient": self.coefficient,
                }
            )
        )

    def kl_loss(self, parameters: Mapping[str, Any], agent_index: int) -> Any:
        """KL(reference || current), per action vector; torch is training-only."""
        import torch

        if type(agent_index) is not int or not 0 <= agent_index < 8:
            raise ValueError("anchor agent index outside roster")
        obs = self.observations[self.active[:, agent_index], agent_index]
        if not len(obs):
            return parameters["b2"].sum() * 0
        reference = self.policy.weights
        old_mean = (
            np.tanh(obs @ reference["w1"][agent_index] + reference["b1"][agent_index])
            @ reference["w2"][agent_index]
            + reference["b2"][agent_index]
        )
        options = {"dtype": parameters["w1"].dtype, "device": parameters["w1"].device}
        x = torch.tensor(obs, **options)
        wanted = torch.tensor(old_mean, **options)
        old_logstd = torch.tensor(reference["log_std"][agent_index].copy(), **options)
        mean = (
            torch.tanh(x @ parameters["w1"] + parameters["b1"]) @ parameters["w2"]
            + parameters["b2"]
        )
        logstd = parameters["log_std"]
        kl = (
            (
                logstd
                - old_logstd
                + (old_logstd.mul(2).exp() + (wanted - mean).square()) / (2 * logstd.mul(2).exp())
                - 0.5
            )
            .sum(dim=1)
            .mean()
        )
        return kl
