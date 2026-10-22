"""Explicit bounded long-horizon SIM velocity stream, not legacy H300.

Uses the exact pinned CPU98->29 model/codec. No resets, action holds or phase
repriming are available. Every20ms must have its own observation/inference.
Long-horizon native stability remains unproven until separately measured.
The legacy G1VelocityPolicy.propose and its H300 proof remain unchanged.
"""

from pathlib import Path

import numpy as np

from rosclaw_soccer.providers.g1.velocity_policy import (
    G1VelocityPolicy,
    observation98,
    phase_delta,
    target29,
)
from rosclaw_soccer.sim.contracts import hash_json


class ContinuousVelocityStream:
    simulation_only = True

    def __init__(
        self,
        model: Path,
        config: Path,
        *,
        expected_model_hash: str,
        expected_config_hash: str,
        horizon_controls: int,
    ) -> None:
        if type(horizon_controls) is not int or not 301 <= horizon_controls <= 6000:
            raise ValueError("new explicitly bounded301..6000 SIM control profile required")
        # Reuse only the strict immutable model/config loader. Never call,
        # mutate, reset or bypass a legacy policy's H300 propose counter.
        loaded = G1VelocityPolicy(
            model,
            config,
            expected_model_hash=expected_model_hash,
            expected_config_hash=expected_config_hash,
        )
        self.params, self.session = loaded.params, loaded.session
        self.horizon_controls = horizon_controls
        self.contract_hash = hash_json(
            dict(
                schema="soccer.g1.continuous_velocity_stream.v1",
                legacy_loader_contract_hash=loaded.contract_hash,
                horizon_controls=horizon_controls,
                control_hz=50,
                legacy_H300_unmodified=True,
                phase_reset_inside_stream=False,
                activation_ceiling="SIM_ONLY",
                hardware_authorized=False,
            )
        )
        self.phase = phase_delta()
        self.last_action = np.zeros(29, np.float32)
        self.last_observation = np.zeros(98, np.float32)
        self.next_frame = self.neural_calls = self.attempted_neural_calls = 0
        self.faulted = False

    def propose(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        angular_velocity_body: np.ndarray,
        command_body: np.ndarray,
        *,
        frame: int,
    ) -> np.ndarray:
        if self.faulted:
            raise ValueError("continuous SIM actor fault latched; no implicit restart")
        try:
            if (
                type(frame) is not int
                or frame != self.next_frame
                or not 0 <= frame < self.horizon_controls
            ):
                raise ValueError(
                    "own consecutive observations within declared SIM horizon required"
                )
            obs, phase = observation98(
                qpos,
                qvel,
                angular_velocity_body,
                command_body,
                self.last_action,
                self.phase,
                self.params,
            )
            self.attempted_neural_calls += 1
            output = self.session.run(["actions"], {"obs": obs[None]})[0]
            self.neural_calls += 1
            if (
                type(output) is not np.ndarray
                or output.shape != (1, 29)
                or output.dtype != np.float32
            ):
                raise ValueError("exact finite CPU98->29 output required")
            target = target29(output[0], self.params)
            self.last_observation, self.last_action = obs.copy(), output[0].copy()
            self.phase = phase
            self.next_frame += 1
            return target
        except Exception:
            self.faulted = True
            raise
