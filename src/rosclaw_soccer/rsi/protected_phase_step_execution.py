"""Compiled causal inference for protected phase motor actor/critic candidates."""

import copy
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi import protected_phase_step_network as network
from rosclaw_soccer.rsi.compiled_step_inference import CompiledStepMotor
from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.contact_motor_primitive import CAP_RAD, SLEW_RAD
from rosclaw_soccer.rsi.step_motor_execution import make_preview as warm_preview
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory
from rosclaw_soccer.rsi.stochastic_step_execution import features_at_frame
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def make_preview(model: dict[str, Any]) -> dict[str, Any]:
    network.validate_model(model)
    policy = make_policy(np.zeros((3, 12)), 0.25, model["model_hash"])
    policy["execution_profile"] = "causal_per_frame_neural_residual"
    policy["step_motor_proof"] = dict(
        schema="soccer.rsi.protected_phase_step_preview.v1",
        model=model,
        decision_start_frame=30,
        nominal_target_is_pre_motor=True,
        force_input="previous_completed_frame",
        qualification="UNQUALIFIED_SIM_COUNTERFACTUAL",
        execution_source_hash=hash_bytes(Path(__file__).read_bytes()),
        promotion_authorized=False,
    )
    policy["protected_phase_motor_proof"] = dict(
        frozen_base_model_hash=model["base_model"]["model_hash"],
        protected_active_frames=model["protected_active_frames"],
        phase_contract=model["phase_contract"],
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledProtectedPhaseMotor:
    def __init__(self, policy: dict[str, Any]) -> None:
        model = copy.deepcopy(policy["step_motor_proof"]["model"])
        if make_preview(model) != policy:
            raise ValueError("protected phase preview differs from sealed model")
        self._planes = network.validate_model(model)
        self._model = model
        self._base = CompiledStepMotor.from_legacy_preview(warm_preview(model["base_model"]))
        self._head = np.asarray(model["actor_readout"], dtype=np.float64).copy()
        self._head.flags.writeable = False
        self._memory = ContactPhaseMemory()
        self._policy_hash = policy["policy_hash"]

    def delta_at_frame(
        self,
        policy: dict[str, Any],
        body: Any,
        *,
        frame: int,
        nominal_target: Any,
        baseline: Any,
        limits: Any,
        previous: Any,
        previous_contact_forces: Any,
    ) -> np.ndarray[Any, Any]:
        if policy.get("policy_hash") != self._policy_hash:
            raise ValueError("compiled protected actor cannot switch commitment")
        phase = self._memory.advance(frame, previous_contact_forces)
        features = features_at_frame(
            body,
            frame=frame,
            nominal_target=nominal_target,
            previous=previous,
            previous_contact_forces=previous_contact_forces,
        )
        base, bounds, prior = [
            np.asarray(v, dtype=np.float64) for v in (baseline, limits, previous)
        ]
        if (
            base.shape != (12,)
            or bounds.shape != (12, 2)
            or prior.shape != (12,)
            or not all(np.isfinite(v).all() for v in (base, bounds, prior))
            or np.any(bounds[:, 0] >= bounds[:, 1])
            or np.max(np.abs(prior)) > CAP_RAD + 1e-5
        ):
            raise ValueError("bounded measured protected motor boundary required")
        if frame < 30:
            if np.any(prior != 0):
                raise ValueError("protected motor cannot predate causal decision boundary")
            return np.zeros(12)
        phi = network.latents(self._model, features[None])[0]
        projected = self._planes[phase].project(phi)
        mean = self._base.raw_mean(features) + 0.05 * np.tanh(self._head[phase] @ projected)
        if not np.isfinite(mean).all():
            raise ValueError("nonfinite protected motor mean")
        desired = CAP_RAD * np.tanh(mean)
        proposed = prior + np.clip(desired - prior, -SLEW_RAD, SLEW_RAD)
        lower = np.maximum(np.minimum(0.0, bounds[:, 0] - base), -CAP_RAD)
        upper = np.minimum(np.maximum(0.0, bounds[:, 1] - base), CAP_RAD)
        return np.asarray(np.clip(proposed, lower, upper), dtype=np.float64)
