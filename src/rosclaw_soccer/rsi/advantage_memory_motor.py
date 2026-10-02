"""A separate SIM-only AWR-inspired actor family over the current NN memory.

Historical PPO/model modules remain artifact-bound and unchanged. This adapter
outputs bounded joint-position residuals, never direct torque or authorization.
"""

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.bounded_advantage_regression as regression_module

from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.current_memory_motor import CompiledCurrentMemoryMotor
from rosclaw_soccer.rsi.current_memory_motor import make_preview as initial_preview
from rosclaw_soccer.rsi.current_memory_motor import validate_model as validate_initial
from rosclaw_soccer.rsi.kernel_guarded_step_network import FLAGS
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.advantage_memory_motor.v1"


def validate_model(model: dict[str, Any]) -> None:
    if (
        model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or model.get("core_regression_source_hash")
        != hash_bytes(Path(regression_module.__file__).read_bytes())
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
        or any(model.get(k) is not False for k in FLAGS)
        or type(model.get("generation")) is not int
        or model["generation"] not in (0, 1)
        or model.get("raw_residual_cap") != 0.2
        or type(model["raw_residual_cap"]) is not float
        or model.get("learning_rate") != 0.0004
        or type(model["learning_rate"]) is not float
    ):
        raise ValueError("sealed bounded SIM-only advantage-memory actor required")
    initial = model["initial_actor"]
    validate_initial(initial)
    if (
        initial["generation"] != 0
        or initial["raw_residual_cap"] != 0.2
        or initial["learning_rate"] != 0.0004
        or model["parent_model_hash"] != initial["parent_model_hash"]
    ):
        raise ValueError("globally exact zero-addition current NN initial actor required")
    if len(model["residual_layers"]) != 3:
        raise ValueError("three bounded finite regression layers required")
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
            raise ValueError("three bounded finite regression layers required")
    receipt = model["learning_receipt"]
    if model["generation"] == 0:
        if receipt is not None or model["residual_layers"] != initial["residual_layers"]:
            raise ValueError("exact zero-addition untrained regression required")
        return
    if (
        not isinstance(receipt, dict)
        or receipt.get("algorithm") != "BOUNDED_ADVANTAGE_WEIGHTED_RESIDUAL_REGRESSION_V1"
        or receipt.get("residual_cap") != 0.2
        or receipt.get("learning_rate") != 0.0004
        or receipt.get("rho") != 0.9
        or receipt.get("temperature") != 0.5
        or receipt.get("maximum_weight") != 20.0
        or receipt.get("critic_kind") != "WHOLE_TRAJECTORY_CROSSFIT_MC_NOT_TD_LAMBDA"
        or receipt.get("critic_crossfit_folds") != 4
        or type(receipt.get("physical_rollout_count")) is not int
        or not 4 <= receipt["physical_rollout_count"] <= 740
        or type(receipt.get("frame_sample_count")) is not int
        or receipt["frame_sample_count"] != receipt["physical_rollout_count"] * 270
        or receipt.get("optimizer_source_hash") != model["core_regression_source_hash"]
        or receipt.get("adapter_source_hash")
        != hash_bytes(Path(__file__).with_name("advantage_memory_learning.py").read_bytes())
        or receipt.get("behavior_model_hash") != initial["baseline"]["base_model"]["model_hash"]
        or receipt.get("protected_memory_hash") != initial["baseline"]["memory"]["memory_hash"]
        or receipt.get("protected_memory_rows")
        != len(initial["baseline"]["memory"]["observations"])
        or receipt.get("protected_anchor_contexts")
        != len(initial["baseline"]["consolidation_manifest"]["records"])
        or receipt.get("frozen_baseline") is not True
        or receipt.get("frozen_guard") is not True
        or any(
            receipt.get(k) is not False
            for k in (
                "physical_batch_verified",
                "distributional_retention_guaranteed",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
        or type(receipt.get("completed_optimizer_steps")) is not int
        or not 1 <= receipt["completed_optimizer_steps"] <= 160
        or any(
            not re.fullmatch(r"sha256:[0-9a-f]{64}", receipt.get(k, ""))
            for k in ("physical_batch_hash", "learner_parent_hash")
        )
        or any(
            type(receipt.get(k)) not in (float, int)
            or not np.isfinite(receipt[k])
            or not 0 <= receipt[k] <= 0.005
            for k in ("exact_mean_conditional_kl", "exact_mean_marginal_kl")
        )
    ):
        raise ValueError("complete bounded advantage regression receipt required")
    if receipt["learner_parent_hash"] != initial_model(initial)["model_hash"]:
        raise ValueError("exact zero-addition learner parent receipt required")
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
        raise ValueError("complete accepted loss history and MC critic required")


def initial_model(actor: dict[str, Any]) -> dict[str, Any]:
    validate_initial(actor)
    model = dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        generation=0,
        initial_actor=copy.deepcopy(actor),
        parent_model_hash=actor["parent_model_hash"],
        residual_layers=copy.deepcopy(actor["residual_layers"]),
        raw_residual_cap=0.2,
        learning_rate=0.0004,
        learning_receipt=None,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        core_regression_source_hash=hash_bytes(Path(regression_module.__file__).read_bytes()),
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
        schema="soccer.rsi.advantage_memory_preview.v1",
        model=model,
        decision_start_frame=30,
        nominal_target_is_pre_motor=True,
        force_input="previous_completed_frame",
        execution_source_hash=model["source_hash"],
        qualification="UNQUALIFIED_SIM_ADVANTAGE_REGRESSION",
        promotion_authorized=False,
    )
    policy["advantage_memory_motor_proof"] = dict(
        parent_model_hash=model["parent_model_hash"],
        memory_hash=model["initial_actor"]["baseline"]["memory"]["memory_hash"],
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledAdvantageMemoryMotor(CompiledCurrentMemoryMotor):
    def __init__(self, policy: dict[str, Any]) -> None:
        model = copy.deepcopy(policy["step_motor_proof"]["model"])
        if make_preview(model) != policy:
            raise ValueError("advantage-memory execution commitment changed")
        super().__init__(initial_preview(model["initial_actor"]))
        self._layers = [
            (np.asarray(v["weight"], dtype=np.float64), np.asarray(v["bias"], dtype=np.float64))
            for v in model["residual_layers"]
        ]
        for w, b in self._layers:
            w.flags.writeable = b.flags.writeable = False
        self._zero = not np.any(self._layers[-1][0]) and not np.any(self._layers[-1][1])
        self._cap = 0.2
        self._policy_hash = policy["policy_hash"]
