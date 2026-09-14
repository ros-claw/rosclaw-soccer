"""Private full-neural kick inference, returning SIM-only raw action proposals.

No simulator stepping, transport, optimizer, trajectory override or activation.
Teacher history, coordinate transforms and physical guards remain caller-owned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.full_kick_checkpoint import load_full_kick_checkpoint
from rosclaw_soccer.training.full_kick_learning import SCHEMA, build_full_kick_actor_critic

_HASH = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class FullKickInferenceProposal:
    agent_id: str
    episode_hash: str
    frame: int
    raw_action: tuple[float, ...]
    learning_active: bool
    observation_hash: str
    contract_hash: str
    activation_ceiling: str = "SIM_ONLY"


class G1FrozenFullKickInference:
    """One private, deterministic 554→29 model with explicit episode sequencing.

    The 547 reference features must come from THIS player's teacher history;
    appended goal/velocity use its pelvis frame. Hash labels cannot establish
    those physical facts: caller-owned evidence must verify them. Training's
    float64 accumulation is retained exactly, with float32 parameters/outputs.
    Output is raw policy space, NOT joint angles, torques or an execution grant.
    This numerical sequence latch is not a daemon safety freeze or motion lease.
    """

    def __init__(
        self,
        weights: Path,
        *,
        expected_actor_hash: str,
        agent_id: str,
        body_hash: str,
        reference_contract_hash: str,
        maximum_episode_frames: int = 400,
    ) -> None:
        if (
            not isinstance(agent_id, str)
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", agent_id) is None
            or any(
                not isinstance(value, str) or _HASH.fullmatch(value) is None
                for value in (body_hash, reference_contract_hash)
            )
            or type(maximum_episode_frames) is not int
            or not 1 <= maximum_episode_frames <= 4096
        ):
            raise ValueError("explicit player/body/reference and bounded episode required")
        parameters, digest = load_full_kick_checkpoint(weights, expected_hash=expected_actor_hash)
        import torch

        reference = {
            key.removeprefix("actor."): value
            for key, value in parameters.items()
            if key.startswith("actor.")
        }
        self._model = build_full_kick_actor_critic(reference)
        self._model.load_state_dict(
            {key: torch.from_numpy(value.copy()) for key, value in parameters.items()}, strict=True
        )
        self._model.eval().requires_grad_(False)
        self._agent_id = agent_id
        self._maximum_frames = maximum_episode_frames
        self._contract_hash = str(
            hash_json(
                dict(
                    schema="soccer.g1_frozen_full_kick_inference.v1",
                    learning_schema=SCHEMA,
                    actor_hash=digest,
                    agent_id=agent_id,
                    body_hash=body_hash,
                    reference_contract_hash=reference_contract_hash,
                    maximum_episode_frames=maximum_episode_frames,
                    output="raw_policy_space_not_joint_targets",
                    activation_ceiling="SIM_ONLY",
                )
            )
        )
        self._active = False
        self._faulted = False
        self._frame = 0
        self._episode_hash = ""
        self._used_episode_hashes: set[str] = set()

    @property
    def contract_hash(self) -> str:
        return self._contract_hash

    def begin_episode(self, *, episode_hash: str) -> None:
        """Explicit fresh numerical sequence; never silently replace an active one."""
        if (
            self._active
            or not isinstance(episode_hash, str)
            or _HASH.fullmatch(episode_hash) is None
            or episode_hash in self._used_episode_hashes
            or len(self._used_episode_hashes) >= 4096
        ):
            raise ValueError("inactive model and a fresh explicit episode hash required")
        self._episode_hash = episode_hash
        self._used_episode_hashes.add(episode_hash)
        self._frame = 0
        self._active = True
        self._faulted = False

    def end_episode(self) -> None:
        """End only this numerical sequence; does not stop physical motion."""
        self._active = False

    def propose(
        self, *, episode_hash: str, frame: int, observation: NDArray[Any]
    ) -> FullKickInferenceProposal:
        if not self._active or self._faulted:
            raise ValueError("fresh declared full-kick episode required")
        try:
            if (
                episode_hash != self._episode_hash
                or type(frame) is not int
                or frame != self._frame
                or not isinstance(observation, np.ndarray)
                or observation.shape != (554,)
                or observation.dtype != np.float32
                or not np.isfinite(observation).all()
                or (np.abs(observation[:547]) > 1e4).any()
                or (np.abs(observation[547:553]) > 1).any()
                or observation[553] not in (0, 1)
            ):
                raise ValueError("matching episode/frame and bounded full554 observation required")
            import torch

            owned = observation.copy()
            with torch.no_grad():
                mean, _ = self._model(torch.from_numpy(owned)[None])
            raw = mean[0].numpy()
            if raw.shape != (29,) or not np.isfinite(raw).all() or (np.abs(raw) > 1e4).any():
                raise ValueError("finite bounded full-kick raw action required")
            result = FullKickInferenceProposal(
                agent_id=self._agent_id,
                episode_hash=self._episode_hash,
                frame=frame,
                raw_action=tuple(float(value) for value in raw),
                learning_active=bool(owned[553]),
                observation_hash=str(hash_json(owned.tolist())),
                contract_hash=self._contract_hash,
            )
        except Exception:
            self._faulted = True
            self._active = False
            raise
        self._frame += 1
        self._active = self._frame < self._maximum_frames
        return result
