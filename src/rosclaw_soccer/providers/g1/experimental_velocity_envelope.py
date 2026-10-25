"""Separate UNQUALIFIED SIM command-envelope experiment for the frozen actor.

Never changes the original loader, command validation, policy, or controller.
The upstream training source suggests a larger domain; it does not prove this
checkpoint reached that curriculum. Native physical qualification is required.
No robot transport or execution permission is exposed here.
"""

import math
from pathlib import Path

import numpy as np

from rosclaw_soccer.providers.g1.velocity_policy import (
    G1VelocityPolicy,
    VelocityPolicyParameters,
    file_hash,
    observation98,
    phase_delta,
    target29,
)
from rosclaw_soccer.sim.contracts import hash_json


def experimental_observation98(
    qpos: np.ndarray,
    qvel: np.ndarray,
    angular_velocity_body: np.ndarray,
    command_body: np.ndarray,
    last_action: np.ndarray,
    phase: np.float32,
    params: VelocityPolicyParameters,
) -> tuple[np.ndarray, np.float32]:
    """Retained body codec with explicitly experimental command/gait fields.

    The old pure codec supplies ONLY unchanged body features and the clock. No
    old controller proposal is called and no old range grant is inferred. The
    experimental command is never clipped to masquerade as a qualified input.
    """
    if (
        type(command_body) is not np.ndarray
        or command_body.shape != (3,)
        or command_body.dtype != np.float32
        or not np.isfinite(command_body).all()
        or not -1 <= command_body[0] <= 2
        or abs(command_body[1]) > 1
        or abs(command_body[2]) > 1
    ):
        raise ValueError("finite experimental x[-1,2]/y[-1,1]/yaw[-1,1] command required")
    obs, next_phase = observation98(
        qpos,
        qvel,
        angular_velocity_body,
        np.zeros(3, np.float32),
        last_action,
        phase,
        params,
    )
    obs[6:9] = command_body
    # Preserve EXACT upstream float32 norm and phase math. Explicit overlap
    # tests compare the entire98 tensor against the untouched qualified codec.
    norm = np.float32(
        np.sqrt(
            np.float32(
                np.float32(command_body[0] * command_body[0] + command_body[1] * command_body[1])
                + command_body[2] * command_body[2]
            )
        )
    )
    if norm >= np.float32(0.1):
        obs[9:11] = [
            math.sin(float(next_phase) * 2 * math.pi),
            math.cos(float(next_phase) * 2 * math.pi),
        ]
    return obs, next_phase


class ExperimentalVelocityEnvelopeStream:
    simulation_only = True
    physically_qualified = False

    def __init__(
        self,
        model: Path,
        config: Path,
        *,
        expected_model_hash: str,
        expected_config_hash: str,
        reference_source: Path,
        expected_reference_hash: str,
        horizon_controls: int,
    ) -> None:
        if (
            type(horizon_controls) is not int
            or not 301 <= horizon_controls <= 1200
            or reference_source.is_symlink()
            or not reference_source.is_file()
            or reference_source.stat().st_size > 262144
            or file_hash(reference_source) != expected_reference_hash
        ):
            raise ValueError("bounded separate SIM horizon and exact training reference required")
        loaded = G1VelocityPolicy(
            model,
            config,
            expected_model_hash=expected_model_hash,
            expected_config_hash=expected_config_hash,
        )
        self.params, self.session = loaded.params, loaded.session
        self.contract_hash = hash_json(
            dict(
                schema="soccer.g1.experimental_velocity_envelope.v1",
                asset_loader_hash=loaded.contract_hash,
                upstream_training_source_hash=expected_reference_hash,
                command_ranges=[[-1, 2], [-1, 1], [-1, 1]],
                checkpoint_curriculum_stage_unknown=True,
                original_qualified_domain_unchanged=True,
                native_qualification_required=True,
                hardware_authorized=False,
                horizon_controls=horizon_controls,
            )
        )
        self.horizon_controls = horizon_controls
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
            raise ValueError("experimental actor fault latched; no implicit reset")
        try:
            if (
                type(frame) is not int
                or frame != self.next_frame
                or not 0 <= frame < self.horizon_controls
            ):
                raise ValueError("own consecutive experimental control frame required")
            obs, phase = experimental_observation98(
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
                raise ValueError("exact float32 experimental actor output required")
            target = target29(output[0], self.params)
            self.last_action, self.last_observation = output[0].copy(), obs.copy()
            self.phase = phase
            self.next_frame += 1
            return target
        except Exception:
            self.faulted = True
            raise
