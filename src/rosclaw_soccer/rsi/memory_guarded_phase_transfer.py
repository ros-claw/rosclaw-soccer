"""Transfer an actual trained phase actor through a complete success-state guard.

No optimizer runs here. Inherited PPO weights are explicitly distinguished from
the added memory gate. The resulting policy remains an unqualified candidate.
"""

import copy
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.anchor_kernel as kernel_module
from rosclaw.growth.anchor_kernel import AnchorKernelGuard

from rosclaw_soccer.rsi import protected_phase_step_network as phase_network
from rosclaw_soccer.rsi.compiled_step_inference import CompiledStepMotor
from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.contact_motor_primitive import CAP_RAD, SLEW_RAD
from rosclaw_soccer.rsi.step_motor_execution import make_preview as warm_preview
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory
from rosclaw_soccer.rsi.stochastic_step_execution import features_at_frame
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.memory_guarded_phase_transfer.v1"


def validate_model(model: dict[str, Any]) -> AnchorKernelGuard:
    if (
        model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or model.get("core_kernel_source_hash")
        != hash_bytes(Path(kernel_module.__file__).read_bytes())
        or model.get("new_optimizer_steps") != 0
        or any(model.get(k) is not False for k in phase_network.FLAGS)
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
    ):
        raise ValueError("unsealed unqualified trained-head transfer")
    phase_network.validate_model(model["phase_model"])
    if (
        model["phase_model"]["generation"] != 1
        or model["inherited_physical_batch_hash"]
        != model["phase_model"]["learning_receipt"]["physical_batch_hash"]
    ):
        raise ValueError("explicit genuinely learned motor head required")
    guard = AnchorKernelGuard.from_dict(model["anchor_guard"])
    if guard.dimension != 134:
        raise ValueError("frozen normalized causal observation guard required")
    return guard


def make_model(phase_model: dict[str, Any], kernel_model: dict[str, Any]) -> dict[str, Any]:
    from rosclaw_soccer.rsi.kernel_guarded_step_network import validate_model as kernel_validate

    phase_network.validate_model(phase_model)
    kernel_validate(kernel_model)
    if phase_model["base_model"] != kernel_model["encoder"]["base_model"]:
        raise ValueError("guard and inherited motor must share a frozen neural parent")
    value = dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        phase_model=copy.deepcopy(phase_model),
        anchor_guard=copy.deepcopy(kernel_model["anchor_guard"]),
        anchor_bank_hash=kernel_model["anchor_bank_hash"],
        new_optimizer_steps=0,
        inherited_physical_batch_hash=phase_model["learning_receipt"]["physical_batch_hash"],
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        core_kernel_source_hash=hash_bytes(Path(kernel_module.__file__).read_bytes()),
        **dict.fromkeys(phase_network.FLAGS, False),
    )
    value["model_hash"] = hash_json(value)
    validate_model(value)
    return value


def make_preview(model: dict[str, Any]) -> dict[str, Any]:
    validate_model(model)
    policy = make_policy(np.zeros((3, 12)), 0.25, model["model_hash"])
    policy["execution_profile"] = "causal_per_frame_neural_residual"
    policy["step_motor_proof"] = dict(
        schema="soccer.rsi.memory_phase_preview.v1",
        model=model,
        decision_start_frame=30,
        nominal_target_is_pre_motor=True,
        force_input="previous_completed_frame",
        execution_source_hash=hash_bytes(Path(__file__).read_bytes()),
        qualification="UNQUALIFIED_SIM_COUNTERFACTUAL",
        promotion_authorized=False,
    )
    policy["memory_phase_motor_proof"] = dict(
        guard_hash=model["anchor_guard"]["guard_hash"],
        inherited_model_hash=model["phase_model"]["model_hash"],
        new_optimizer_steps=0,
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledMemoryPhaseMotor:
    def __init__(self, policy: dict[str, Any]) -> None:
        model = copy.deepcopy(policy["step_motor_proof"]["model"])
        if make_preview(model) != policy:
            raise ValueError("memory phase transfer execution binding changed")
        self._guard = validate_model(model)
        self._model = model["phase_model"]
        self._planes = phase_network.validate_model(self._model)
        self._warm = CompiledStepMotor.from_legacy_preview(warm_preview(self._model["base_model"]))
        self._head = np.asarray(self._model["actor_readout"])
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
    ) -> Any:
        if policy.get("policy_hash") != self._policy_hash:
            raise ValueError("running memory motor cannot switch commitment")
        phase = self._memory.advance(frame, previous_contact_forces)
        x = features_at_frame(
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
            raise ValueError("bounded actual motor boundary required")
        if frame < 30:
            if np.any(prior != 0):
                raise ValueError("memory motor cannot predate causal decision boundary")
            return np.zeros(12)
        phi = phase_network.latents(self._model, x[None])[0]
        mean = self._warm.raw_mean(x) + 0.05 * self._guard.gate(phi[:134]) * np.tanh(
            self._head[phase] @ self._planes[phase].project(phi)
        )
        if not np.isfinite(mean).all():
            raise ValueError("nonfinite inherited motor mean")
        proposed = prior + np.clip(CAP_RAD * np.tanh(mean) - prior, -SLEW_RAD, SLEW_RAD)
        lower = np.maximum(np.minimum(0.0, bounds[:, 0] - base), -CAP_RAD)
        upper = np.minimum(np.maximum(0.0, bounds[:, 1] - base), CAP_RAD)
        return np.asarray(np.clip(proposed, lower, upper), dtype=np.float64)
