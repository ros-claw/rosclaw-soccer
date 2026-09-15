"""Learn a local residual skill space around a frozen goal-conditioned receiver."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.approach_router import load_bounded_reference_parameters
from rosclaw_soccer.providers.g1.recurrent_receiver import G1RecurrentReceiver
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic
from rosclaw_soccer.training.motor_synergy import MotorSynergyBasis


class G1SynergyReceiver(G1RecurrentReceiver):
    def __init__(
        self,
        weights: Path,
        *,
        latent_weights: Path,
        expected_latent_hash: str,
        synergy: MotorSynergyBasis,
        seed: int,
        explore: bool = False,
        **kwargs: Any,
    ) -> None:
        import torch

        if (
            not isinstance(synergy, MotorSynergyBasis)
            or len(synergy.matrix) != 29
            or type(seed) is not int
            or not 0 <= seed < 2**32
            or type(explore) is not bool
            or kwargs.get("observation_contract")
            != "recurrent_receiver_followup_target_138_float32.v3"
        ):
            raise ValueError("explicit G1 goal-conditioned motor-synergy contract required")
        super().__init__(weights, **kwargs)
        self._synergy = synergy
        self._synergy_hash = synergy.basis_hash
        self._latent = build_ball_residual_actor_critic(observation_size=138, action_size=3)
        shapes = {k: tuple(v.shape) for k, v in self._latent.state_dict().items()}
        parameters, digest = load_bounded_reference_parameters(
            latent_weights, expected_latent_hash, shapes
        )
        self._latent.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in parameters.items()})
        self._latent.requires_grad_(False)
        self._latent.eval()
        self._rng = np.random.default_rng(seed)
        self._explore = explore
        self.samples: list[dict[str, np.ndarray]] = []
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "soccer.g1_synergy_receiver.v1",
                    "frozen_receiver": self.contract_hash,
                    "latent_weights": digest,
                    "basis": self._synergy_hash,
                    "explore": explore,
                    "seed": seed,
                    "minimum_log_std": -2.5,
                    "maximum_log_std": -0.3,
                    "activation_ceiling": "SIM_ONLY",
                }
            )
        )

    def _select_raw_action(self, features: Any, mean: Any, value: Any) -> Any:
        import torch

        if self._synergy.basis_hash != self._synergy_hash:
            raise ValueError("motor basis changed inside a bound skill")
        latent_mean, latent_value = self._latent(features)
        distribution: Any = torch.distributions.Normal(
            latent_mean, self._latent.logstd.clamp(-2.5, -0.3).exp()
        )
        noise = torch.tensor(self._rng.standard_normal((1, 3)), dtype=torch.float32)
        raw = latent_mean + distribution.scale * noise if self._explore else latent_mean
        self.samples.append(
            {
                "obs": features[0].detach().cpu().numpy().copy(),
                "raw": raw[0].detach().cpu().numpy().copy(),
                "logp": distribution.log_prob(raw).sum(1)[0].detach().cpu().numpy().copy(),
                "value": latent_value[0].detach().cpu().numpy().copy(),
            }
        )
        offset = self._synergy.decode(raw[0].detach().cpu().numpy())
        # Parent filter and joint/torque guards remain the sole output path.
        return mean + torch.tensor(offset, dtype=mean.dtype, device=mean.device)[None]

    def sampled_rollout(self) -> dict[str, np.ndarray]:
        if not self.samples:
            raise ValueError("motor synergy has no admitted samples")
        return {k: np.stack([row[k] for row in self.samples]) for k in self.samples[0]}
