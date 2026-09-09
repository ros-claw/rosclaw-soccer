"""Frozen learned receiving residual over a live player's recurrent foundation.

Owns only a 100-tick residual filter. Never owns/restarts the LSTM, supplies
navigation, steps physics, selects a role, or grants skill admission/promotion.
The caller supplies same-player/frame immutable foundation observations.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.approach_router import load_bounded_reference_parameters
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation, TeamMotorTarget
from rosclaw_soccer.training.ball_residual import (
    advance_ball_residual,
    build_ball_residual_actor_critic,
)


class G1RecurrentReceiver:
    """Per-player, bounded frozen actor. A proposal fault stays latched."""

    def __init__(
        self,
        weights: Path,
        *,
        agent_id: str,
        expected_actor_hash: str,
        foundation_hash: str,
        foundation_config_hash: str,
    ) -> None:
        import torch

        if (
            not isinstance(agent_id, str)
            or re.fullmatch(r"[a-z][a-z0-9_.:-]{0,127}", agent_id) is None
            or not isinstance(foundation_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", foundation_hash) is None
            or not isinstance(foundation_config_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", foundation_config_hash) is None
        ):
            raise ValueError("explicit player and frozen foundation hash required")
        self._actor = build_ball_residual_actor_critic()
        shapes = {k: tuple(v.shape) for k, v in self._actor.state_dict().items()}
        parameters, digest = load_bounded_reference_parameters(weights, expected_actor_hash, shapes)
        self._actor.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in parameters.items()})
        self._actor.requires_grad_(False)
        self._actor.eval()
        self.agent_id, self.foundation_hash = agent_id, foundation_hash
        self.foundation_config_hash = foundation_config_hash
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "soccer.g1_recurrent_receiver.v1",
                    "actor_hash": digest,
                    "foundation_artifact_hash": foundation_hash,
                    "foundation_configuration_hash": foundation_config_hash,
                    "agent_id": agent_id,
                    "observation": "recurrent_receiver_133_float32.v1",
                    "episode_frames": 100,
                    "control_dt_sec": 0.02,
                    "maximum_offset_rad": 0.25,
                    "maximum_step_rad": 0.025,
                    "smoothing": 0.2,
                    "activation_ceiling": "SIM_ONLY",
                }
            )
        )
        self._start_frame: int | None = None
        self._start_time = 0.0
        self._next_frame = 0
        self._faulted = False
        self._previous = torch.zeros((1, 29))
        self.last_observation: Any = None

    def _validate(self, observation: TeamMotorObservation) -> None:
        if (
            not isinstance(observation, TeamMotorObservation)
            or observation.agent_id != self.agent_id
            or observation.foundation is None
            or observation.foundation.policy_hash != self.foundation_hash
            or observation.foundation.configuration_hash != self.foundation_config_hash
        ):
            raise ValueError("same-player frozen foundation observation required")

    def begin_skill(self, observation: TeamMotorObservation) -> None:
        """Declare a new bounded proposal skill; never reset body or LSTM state.

        No implicit retry after a fault or restart during/after this instance's
        skill. A new per-player instance is required at a new admission boundary.
        """
        self._validate(observation)
        if self._faulted or self._start_frame is not None:
            raise ValueError("receiver instance already used or faulted")
        self._start_frame = observation.frame
        self._start_time = observation.time_sec

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget:
        import torch

        if self._faulted or self._start_frame is None:
            raise ValueError("explicit unfaulted receiver skill required")
        try:
            self._validate(observation)
            frame = observation.frame - self._start_frame
            if (
                frame != self._next_frame
                or not 0 <= frame < 100
                or abs(observation.time_sec - self._start_time - frame * 0.02) > 1e-6
            ):
                raise ValueError("consecutive 50 Hz receiver frames required")
            assert observation.foundation is not None
            foundation = observation.foundation
            p = torch.tensor(observation.qpos, dtype=torch.float32)[None]
            w = torch.tensor(observation.qvel, dtype=torch.float32)[None]
            if abs(float(torch.linalg.vector_norm(p[0, 3:7])) - 1.0) > 1e-4:
                raise ValueError("normalized receiver root quaternion required")
            default = torch.tensor(foundation.default_angles, dtype=torch.float32)
            base = torch.tensor(foundation.target.target_rad, dtype=torch.float32)[None]
            qw, qx, qy, qz = p[:, 3:7].unbind(1)
            gravity = torch.stack(
                (
                    2 * (-qz * qx + qw * qy),
                    -2 * (qz * qy + qw * qx),
                    1 - 2 * (qw * qw + qz * qz),
                ),
                1,
            )
            phase = torch.full((1, 1), frame / 100)
            features = torch.cat(
                (
                    p[:, 7:36] - default,
                    w[:, 6:35] * 0.1,
                    gravity,
                    w[:, :3],
                    w[:, 3:6] * 0.2,
                    p[:, 36:39] - p[:, :3],
                    w[:, 35:38],
                    torch.sin(phase * 2 * np.pi),
                    torch.cos(phase * 2 * np.pi),
                    self._previous,
                    base - default,
                ),
                1,
            )
            if not bool(torch.isfinite(features).all()):
                raise ValueError("receiver feature conversion overflow")
            features = features.clamp(-10, 10)
            with torch.no_grad():
                mean, _ = self._actor(features)
                residual = advance_ball_residual(mean, self._previous)
                target = base + residual
            proposal = TeamMotorTarget(
                tuple(float(x) for x in target[0]), foundation.target.kp, foundation.target.kd
            )
        except (ValueError, TypeError, OverflowError, FloatingPointError, RuntimeError) as exc:
            self._faulted = True
            if isinstance(exc, (RuntimeError, OverflowError)):
                raise ValueError("receiver inference failed; candidate remains faulted") from exc
            raise
        self._previous = residual
        self.last_observation = features.clone()
        self._next_frame += 1
        return proposal
