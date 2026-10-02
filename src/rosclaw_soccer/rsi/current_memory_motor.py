"""Current-parent anchors protect plastic AR motor layers DURING learning.

This is a new model family. Historical smooth/consolidated actors stay intact.
The residual cap is a raw latent cap, not joint torque or physical authority.
"""

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.correlated_residual_gradient as gradient_module
from rosclaw.growth.anchor_kernel import AnchorKernelGuard

from rosclaw_soccer.rsi.consolidated_smooth_motor import validate_model as validate_baseline
from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.kernel_guarded_step_network import FLAGS
from rosclaw_soccer.rsi.smooth_memory_motor import CompiledSmoothMemoryMotor
from rosclaw_soccer.rsi.smooth_memory_motor import make_preview as baseline_preview
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.current_memory_guarded_motor.v1"


def validate_model(model: dict[str, Any]) -> None:
    if (
        model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or model.get("core_gradient_source_hash")
        != hash_bytes(Path(gradient_module.__file__).read_bytes())
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
        or any(model.get(k) is not False for k in FLAGS)
        or type(model.get("generation")) is not int
        or not 0 <= model["generation"] <= 32
        or type(model.get("raw_residual_cap")) not in (float, int)
        or model["raw_residual_cap"] not in (0.05, 0.2)
        or type(model.get("learning_rate")) not in (float, int)
        or model["learning_rate"] not in (1e-4, 4e-4)
        or model.get("guard_bandwidth") != 1e-4
        or model.get("protection_scope")
        != "ALL_CURRENT_PARENT_SUCCESS_ANCHORS_DURING_TRAINING_AND_EXECUTION"
    ):
        raise ValueError("sealed bounded SIM-only current-memory actor required")
    validate_baseline(model["baseline"])
    base = model["baseline"]["base_model"]
    if base["generation"] != 0 or model["parent_model_hash"] != base["parent_model_hash"]:
        raise ValueError("globally exact zero-addition qualified NN baseline required")
    if len(model["residual_layers"]) != 3:
        raise ValueError("three finite aligned plastic layers required")
    for layer, shape in zip(
        model["residual_layers"], ((128, 135), (128, 128), (12, 128)), strict=True
    ):
        w, b = np.asarray(layer["weight"]), np.asarray(layer["bias"])
        if (
            w.shape != shape
            or b.shape != (shape[0],)
            or not np.isfinite(w).all()
            or not np.isfinite(b).all()
        ):
            raise ValueError("three finite aligned plastic layers required")
    receipt = model["learning_receipt"]
    if model["generation"] == 0:
        last = model["residual_layers"][-1]
        if receipt is not None or np.any(last["weight"]) or np.any(last["bias"]):
            raise ValueError("globally exact zero plastic initialization required")
    elif (
        not isinstance(receipt, dict)
        or receipt.get("algorithm") != "BOUNDED_CORRELATED_RESIDUAL_PPO_V1"
        or receipt.get("residual_cap") != model["raw_residual_cap"]
        or receipt.get("learning_rate") != model["learning_rate"]
        or receipt.get("rho") != 0.9
        or receipt.get("frozen_baseline") is not True
        or receipt.get("frozen_guard") is not True
        or receipt.get("physical_batch_verified") is not False
        or receipt.get("distributional_retention_guaranteed") is not False
        or type(receipt.get("completed_optimizer_steps")) is not int
        or not 1 <= receipt["completed_optimizer_steps"] <= 160
        or any(
            not re.fullmatch(r"sha256:[0-9a-f]{64}", receipt.get(k, ""))
            for k in ("physical_batch_hash", "learner_parent_hash", "optimizer_source_hash")
        )
        or receipt["optimizer_source_hash"] != model["core_gradient_source_hash"]
        or receipt.get("behavior_model_hash") != base["model_hash"]
        or receipt.get("behavior_mean_equivalence")
        != "GLOBALLY_EXACT_ZERO_ADDITION_TO_SAME_FROZEN_NN"
        or receipt.get("protected_memory_hash") != model["baseline"]["memory"]["memory_hash"]
        or receipt.get("protected_memory_rows") != len(model["baseline"]["memory"]["observations"])
        or receipt.get("protected_anchor_contexts")
        != len(model["baseline"]["consolidation_manifest"]["records"])
        or receipt.get("adapter_source_hash")
        != hash_bytes(Path(__file__).with_name("current_memory_learning.py").read_bytes())
        or any(
            type(receipt.get(k)) not in (float, int)
            or not np.isfinite(receipt[k])
            or not 0 <= receipt[k] <= 0.005
            for k in ("exact_mean_conditional_kl", "exact_mean_marginal_kl")
        )
        or any(receipt.get(k) is not False for k in ("promotion_authorized", "hardware_authorized"))
    ):
        raise ValueError("bounded correlated learning receipt required")
    if model["generation"] > 0:
        history = receipt.get("full_batch_loss_history")
        critic = np.asarray(model.get("critic_readout"))
        if (
            not isinstance(history, list)
            or len(history) != receipt["completed_optimizer_steps"] + 1
            or any(type(v) not in (float, int) or not np.isfinite(v) for v in history)
            or any(a <= b for a, b in zip(history, history[1:], strict=False))
            or critic.shape != (3, 512)
            or not np.isfinite(critic).all()
        ):
            raise ValueError("complete finite accepted-step history and critic required")


def initial_model(
    baseline: dict[str, Any], *, cap: float = 0.2, learning_rate: float = 4e-4
) -> dict[str, Any]:
    validate_baseline(baseline)
    base = baseline["base_model"]
    model = dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        generation=0,
        baseline=copy.deepcopy(baseline),
        parent_model_hash=base["parent_model_hash"],
        residual_layers=copy.deepcopy(base["residual_layers"]),
        raw_residual_cap=cap,
        learning_rate=learning_rate,
        guard_bandwidth=1e-4,
        protection_scope="ALL_CURRENT_PARENT_SUCCESS_ANCHORS_DURING_TRAINING_AND_EXECUTION",
        learning_receipt=None,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        core_gradient_source_hash=hash_bytes(Path(gradient_module.__file__).read_bytes()),
        **dict.fromkeys(FLAGS, False),
    )
    model["model_hash"] = hash_json(model)
    validate_model(model)
    return model


def make_preview(model: dict[str, Any]) -> dict[str, Any]:
    validate_model(model)
    policy = make_policy(np.zeros((3, 12)), 0.25, model["model_hash"])
    policy["execution_profile"] = "causal_per_frame_neural_residual"
    policy["step_motor_proof"] = dict(
        schema="soccer.rsi.current_memory_guarded_preview.v1",
        model=model,
        decision_start_frame=30,
        nominal_target_is_pre_motor=True,
        force_input="previous_completed_frame",
        execution_source_hash=model["source_hash"],
        qualification="UNQUALIFIED_SIM_CURRENT_MEMORY_LEARNING",
        promotion_authorized=False,
    )
    policy["current_memory_motor_proof"] = dict(
        parent_model_hash=model["parent_model_hash"],
        memory_hash=model["baseline"]["memory"]["memory_hash"],
        protection_scope=model["protection_scope"],
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledCurrentMemoryMotor(CompiledSmoothMemoryMotor):
    def __init__(self, policy: dict[str, Any]) -> None:
        model = copy.deepcopy(policy["step_motor_proof"]["model"])
        if make_preview(model) != policy:
            raise ValueError("current-memory motor execution commitment changed")
        super().__init__(baseline_preview(model["baseline"]["base_model"]))
        self._guard = AnchorKernelGuard(model["baseline"]["memory"]["observations"], bandwidth=1e-4)
        self._layers = [
            (np.asarray(v["weight"], dtype=np.float64), np.asarray(v["bias"], dtype=np.float64))
            for v in model["residual_layers"]
        ]
        for w, b in self._layers:
            w.flags.writeable = b.flags.writeable = False
        self._cap = model["raw_residual_cap"]
        self._zero = not np.any(self._layers[-1][0]) and not np.any(self._layers[-1][1])
        self._sampling = None
        self._policy_hash = policy["policy_hash"]

    def raw_mean(self, observation: Any, phase: int) -> Any:
        mean = self._parent.raw_mean(observation, phase)
        if self._zero:
            return mean
        context = np.concatenate((self.features(observation)[:134], [phase]))
        gate = self._guard.gate(context)
        if gate == 0:
            return mean
        hidden = context
        for w, b in self._layers:
            hidden = np.tanh(w @ hidden + b)
        result = mean + self._cap * gate * hidden
        if not np.isfinite(result).all():
            raise ValueError("nonfinite current-memory motor mean")
        return result
