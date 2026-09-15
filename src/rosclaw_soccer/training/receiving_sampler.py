"""Seeded, recorded on-policy receiving exploration; never a deployed actor.

The world still owns admission, contact, safety and all simulation stepping.
Gaussian likelihoods refer to raw samples before the existing bounded filter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.recurrent_receiver import G1RecurrentReceiver
from rosclaw_soccer.sim.contracts import hash_json


class ReceivingSampler(G1RecurrentReceiver):
    def __init__(
        self,
        weights: Path,
        *,
        seed: int,
        agent_id: str,
        expected_actor_hash: str,
        foundation_hash: str,
        foundation_config_hash: str,
        observation_contract: str = "recurrent_receiver_133_float32.v1",
        episode_frames: int = 100,
        followup_target_position_m: tuple[float, float, float] | None = None,
    ) -> None:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("bounded explicit receiving exploration seed required")
        super().__init__(
            weights,
            agent_id=agent_id,
            expected_actor_hash=expected_actor_hash,
            foundation_hash=foundation_hash,
            foundation_config_hash=foundation_config_hash,
            observation_contract=observation_contract,
            episode_frames=episode_frames,
            followup_target_position_m=followup_target_position_m,
        )
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "soccer.receiving_sampler.v1",
                    "frozen_receiver": self.contract_hash,
                    "seed": seed,
                    "minimum_log_std": -2.5,
                    "maximum_log_std": -0.3,
                    "activation_ceiling": "SIM_ONLY",
                    "training_only": True,
                }
            )
        )
        self._rng = np.random.default_rng(seed)
        self.samples: list[dict[str, np.ndarray]] = []

    def _select_raw_action(self, features: Any, mean: Any, value: Any) -> Any:
        import torch

        distribution: Any = torch.distributions.Normal(
            mean, self._actor.logstd.clamp(-2.5, -0.3).exp()
        )
        noise = torch.tensor(
            self._rng.standard_normal(tuple(mean.shape)), dtype=mean.dtype, device=mean.device
        )
        raw = mean + distribution.scale * noise
        self.samples.append(
            {
                "obs": features[0].detach().cpu().numpy().copy(),
                "raw": raw[0].detach().cpu().numpy().copy(),
                "logp": distribution.log_prob(raw).sum(1)[0].detach().cpu().numpy().copy(),
                "value": value[0].detach().cpu().numpy().copy(),
            }
        )
        return raw

    def sampled_rollout(self) -> dict[str, np.ndarray]:
        """Independent copies; these are actions, not physical reception labels."""
        if not self.samples:
            raise ValueError("receiver produced no admitted learning samples")
        return {key: np.stack([sample[key] for sample in self.samples]) for key in self.samples[0]}
