"""SIM_ONLY episode-coherent exploration of post-contact motor memory."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.receiving_adaptive_phase_motor import ReceivingAdaptivePhaseMotor
from rosclaw_soccer.sim.contracts import hash_json


@dataclass
class ReceivingLatentPhaseMotor(ReceivingAdaptivePhaseMotor):
    """One bounded 12D latent offset per episode, held through the contact phase."""

    latent_std: float = 0.0
    latent_seed: int = 0
    latent_offset: tuple[float, ...] = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.post_exploration_std != 0.0
            or type(self.latent_std) is not float
            or not math.isfinite(self.latent_std)
            or not 0 <= self.latent_std <= 0.5
            or type(self.latent_seed) is not int
            or not 0 <= self.latent_seed < 2**31
        ):
            raise ValueError("bounded episode-coherent latent exploration required")
        rng = np.random.default_rng(self.latent_seed)
        self.latent_offset = tuple(
            float(x) for x in np.clip(rng.normal(0.0, self.latent_std, 12), -1.2, 1.2)
        )
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_latent_phase_motor.v1",
                "parent_contract_hash": self.contract_hash,
                "latent_std": self.latent_std,
                "latent_seed": self.latent_seed,
                "latent_offset": self.latent_offset,
                "selection": "one_episode_latent_held_through_contact",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def _post_logits(self, features: tuple[float, ...]) -> np.ndarray:
        return super()._post_logits(features) + np.asarray(self.latent_offset)
