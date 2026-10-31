"""Causal SIM-only navigation exploration; proposals are not applied-action evidence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.approach_router import load_bounded_reference_parameters
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.navigation_option import NavigationDelta, NavigationObservation
from rosclaw_soccer.training.local_navigation import (
    build_local_navigation_actor_critic,
    local_navigation_delta,
    local_navigation_features,
)


class LiveNavigationSampler:
    """Sample a frozen 39/3 actor under an explicit player-intent gate.

    The world still projects proposals through clearance and arbitration. The
    sampler records what was proposed, not whether it affected the trajectory.
    A caller must independently verify applied actions and on-policy timing.
    """

    activation_ceiling = "SIM_ONLY"

    def __init__(
        self,
        weights: Path,
        *,
        agent_id: str,
        actor_hash: str,
        foundation_hash: str,
        foundation_config_hash: str,
        seed: int,
        gate: str = "committed_receiver",
    ) -> None:
        import torch

        if (
            not isinstance(agent_id, str)
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", agent_id) is None
            or any(
                not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
                for value in (actor_hash, foundation_hash, foundation_config_hash)
            )
            or type(seed) is not int
            or not 0 <= seed < 2**32
            or type(gate) is not str
            or gate not in ("committed_receiver", "ball_intent")
        ):
            raise ValueError("explicit player, model identities and unsigned seed required")
        actor = build_local_navigation_actor_critic()
        params, digest = load_bounded_reference_parameters(
            weights,
            actor_hash,
            {key: tuple(value.shape) for key, value in actor.state_dict().items()},
        )
        actor.load_state_dict(
            {key: torch.from_numpy(value.copy()) for key, value in params.items()}
        )
        actor.requires_grad_(False)
        actor.eval()
        self._actor = actor
        self._rng = np.random.default_rng(seed)
        self.agent_id = agent_id
        self.foundation_hash = foundation_hash
        self.foundation_config_hash = foundation_config_hash
        self.gate = gate
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "soccer.live_navigation_sampler.v1",
                    "agent_id": agent_id,
                    "actor_hash": digest,
                    "foundation_hash": foundation_hash,
                    "foundation_config_hash": foundation_config_hash,
                    "seed": seed,
                    "gate": gate,
                    "observation": "soccer.local_navigation_world_heading_39.v1",
                    "raw_action": "gaussian_3_logstd_-2.5_-0.3",
                    "activation_ceiling": self.activation_ceiling,
                    "training_only": True,
                }
            )
        )
        self.samples: list[dict[str, Any]] = []

    def propose(self, observation: NavigationObservation) -> NavigationDelta | None:
        import torch

        if (
            not isinstance(observation, NavigationObservation)
            or observation.agent_id != self.agent_id
        ):
            raise ValueError("same-player navigation observation required")
        if not self._eligible(observation):
            return None
        features = local_navigation_features(observation)
        with torch.no_grad():
            mean, value = self._actor(torch.from_numpy(features[None]))
            normal: Any = torch.distributions.Normal(
                mean, self._actor.logstd.clamp(-2.5, -0.3).exp()
            )
            noise = torch.tensor(self._rng.standard_normal(tuple(mean.shape)), dtype=mean.dtype)
            raw = mean + normal.scale * noise
            logp = normal.log_prob(raw).sum(1)
        raw_np = raw[0].cpu().numpy().copy()
        delta = local_navigation_delta(raw_np)
        self.samples.append(
            {
                "frame": observation.frame,
                "obs": features.copy(),
                "raw": raw_np,
                "logp": float(logp[0]),
                "value": float(value[0]),
                "proposed_delta": np.asarray(delta, dtype=np.float32),
            }
        )
        return NavigationDelta(observation.agent_id, observation.frame, observation.time_sec, delta)

    def _eligible(self, observation: NavigationObservation) -> bool:
        """Opt-in intent exploration never upgrades permission or success."""
        return bool(
            observation.committed_receiver
            or self.gate == "ball_intent"
            and observation.intent in {"receive", "intercept", "press"}
        )


class DeterministicNavigationCandidate(LiveNavigationSampler):
    """Frozen mean-action research candidate, not a promoted player skill."""

    def __init__(
        self,
        weights: Path,
        *,
        agent_id: str,
        actor_hash: str,
        foundation_hash: str,
        foundation_config_hash: str,
        gate: str = "committed_receiver",
    ) -> None:
        super().__init__(
            weights,
            agent_id=agent_id,
            actor_hash=actor_hash,
            foundation_hash=foundation_hash,
            foundation_config_hash=foundation_config_hash,
            seed=0,
            gate=gate,
        )
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "soccer.deterministic_navigation_candidate.v1",
                    "agent_id": agent_id,
                    "actor_hash": actor_hash,
                    "foundation_hash": foundation_hash,
                    "foundation_config_hash": foundation_config_hash,
                    "gate": gate,
                    "observation": "soccer.local_navigation_world_heading_39.v1",
                    "action": "bounded_gaussian_mean",
                    "activation_ceiling": self.activation_ceiling,
                    "promotion_eligible": False,
                }
            )
        )

    def propose(self, observation: NavigationObservation) -> NavigationDelta | None:
        import torch

        if (
            not isinstance(observation, NavigationObservation)
            or observation.agent_id != self.agent_id
        ):
            raise ValueError("same-player navigation observation required")
        if not self._eligible(observation):
            return None
        features = local_navigation_features(observation)
        with torch.no_grad():
            mean, _ = self._actor(torch.from_numpy(features[None]))
        return NavigationDelta(
            observation.agent_id,
            observation.frame,
            observation.time_sec,
            local_navigation_delta(mean[0].cpu().numpy()),
        )
