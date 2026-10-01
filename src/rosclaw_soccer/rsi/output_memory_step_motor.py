"""Later-parent output memory with a fully plastic bounded residual MLP.

The foundation and parent are frozen. All layers of the new residual can learn;
the frozen metric contains causal phase but no course identity or outcome.
Remembered outputs are local training constraints, not trajectory guarantees.
"""

import copy
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import rosclaw.growth.anchor_output_memory as memory_module
from rosclaw.growth.anchor_output_memory import AnchorOutputMemory

from rosclaw_soccer.rsi import step_motor_features, step_motor_phase_context
from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.kernel_guarded_step_execution import CompiledKernelStepMotor
from rosclaw_soccer.rsi.kernel_guarded_step_execution import make_preview as parent_preview
from rosclaw_soccer.rsi.kernel_guarded_step_network import FLAGS
from rosclaw_soccer.rsi.kernel_guarded_step_network import validate_model as parent_validate
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.output_memory_step_motor.v1"
SAMPLING_SCHEMA = "soccer.rsi.output_memory_step_sampling.v1"


def encoder_identity(parent: dict[str, Any]) -> str:
    return hash_json(
        dict(
            schema="soccer.rsi.frozen_motor_memory_context.v1",
            warm_model_hash=parent["encoder"]["base_model"]["model_hash"],
            feature_source_hash=hash_bytes(Path(step_motor_features.__file__).read_bytes()),
            phase_source_hash=hash_bytes(Path(step_motor_phase_context.__file__).read_bytes()),
            metric="concat(clipped_normalized_134,causal_phase_index)",
        )
    )


def validate_model(model: dict[str, Any]) -> AnchorOutputMemory:
    if (
        model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or model.get("core_memory_source_hash")
        != hash_bytes(Path(memory_module.__file__).read_bytes())
        or any(model.get(k) is not False for k in FLAGS)
        or type(model.get("generation")) is not int
        or not 0 <= model["generation"] <= 32
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
    ):
        raise ValueError("sealed SIM-only output-memory motor required")
    parent_validate(model["frozen_parent"])
    memory = AnchorOutputMemory.from_dict(model["output_memory"])
    if (
        memory.encoder_hash != encoder_identity(model["frozen_parent"])
        or memory.parent_policy_hash != model["frozen_parent"]["model_hash"]
        or memory.output_dimension != 12
        or np.asarray(model["output_memory"]["observations"]).shape[1] != 135
        or model["raw_residual_cap"] != 0.05
        or model["parent_model_hash"] != model["frozen_parent"]["model_hash"]
        or model["output_memory"]["bandwidth"] != 1e-4
        or len(model["residual_layers"]) != 3
    ):
        raise ValueError("complete frozen current-parent causal output memory required")
    for layer, shape in zip(
        model["residual_layers"], ((128, 135), (128, 128), (12, 128)), strict=True
    ):
        weight, bias = np.asarray(layer["weight"]), np.asarray(layer["bias"])
        if (
            weight.shape != shape
            or bias.shape != (shape[0],)
            or not np.isfinite(weight).all()
            or not np.isfinite(bias).all()
        ):
            raise ValueError("finite aligned residual network required")
    if model["generation"] == 0:
        last = model["residual_layers"][-1]
        if (
            model["learning_receipt"] is not None
            or np.any(np.asarray(last["weight"]))
            or np.any(np.asarray(last["bias"]))
        ):
            raise ValueError("zero residual initialization required")
    else:
        receipt = model["learning_receipt"]
        if (
            not isinstance(receipt, dict)
            or receipt.get("algorithm") != "OUTPUT_MEMORY_MLP_PPO_MC_TERMINAL"
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", receipt.get("physical_batch_hash", ""))
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", receipt.get("learner_parent_hash", ""))
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", receipt.get("optimizer_source_hash", ""))
            or type(receipt.get("completed_optimizer_steps")) is not int
            or not 1 <= receipt["completed_optimizer_steps"] <= 160
            or receipt.get("all_residual_layers_trainable") is not True
            or receipt.get("frozen_parent") is not True
            or receipt.get("distributional_retention_guaranteed") is not False
            or type(receipt.get("exact_mean_latent_kl")) not in (float, int)
            or not np.isfinite(receipt["exact_mean_latent_kl"])
            or not 0 <= receipt["exact_mean_latent_kl"] <= 0.005
            or any(
                receipt.get(k) is not False for k in ("promotion_authorized", "hardware_authorized")
            )
        ):
            raise ValueError("actual bounded output-memory learning receipt required")
    return memory


def initial_model(parent: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(202610342)
    layers = []
    for out_dim, in_dim in ((128, 135), (128, 128), (12, 128)):
        weight = (
            np.zeros((out_dim, in_dim))
            if out_dim == 12
            else rng.normal(size=(out_dim, in_dim)) / np.sqrt(in_dim)
        )
        layers.append(dict(weight=weight.tolist(), bias=np.zeros(out_dim).tolist()))
    model = dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        generation=0,
        parent_model_hash=parent["model_hash"],
        frozen_parent=copy.deepcopy(parent),
        output_memory=copy.deepcopy(memory),
        residual_layers=layers,
        raw_residual_cap=0.05,
        learning_receipt=None,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        core_memory_source_hash=hash_bytes(Path(memory_module.__file__).read_bytes()),
        **dict.fromkeys(FLAGS, False),
    )
    model["model_hash"] = hash_json(model)
    validate_model(model)
    return model


def make_sampling_view(model: dict[str, Any], *, seed: int, std: float) -> dict[str, Any]:
    validate_model(model)
    if (
        type(seed) is not int
        or not 0 <= seed < 2**32
        or type(std) not in (float, int)
        or not np.isfinite(std)
        or not 0.01 <= std <= 0.15
    ):
        raise ValueError("bounded explicit simulation exploration required")
    view = dict(
        schema=SAMPLING_SCHEMA,
        mean_model=copy.deepcopy(model),
        seed=seed,
        std_raw=float(std),
        activation_ceiling="SIM_ONLY",
        training_only=True,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        **dict.fromkeys(FLAGS, False),
    )
    view["model_hash"] = hash_json(view)
    return view


def mean_model(model: dict[str, Any]) -> dict[str, Any]:
    if model.get("schema") == SAMPLING_SCHEMA:
        if (
            model.get("model_hash")
            != hash_json({k: v for k, v in model.items() if k != "model_hash"})
            or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
            or model.get("activation_ceiling") != "SIM_ONLY"
            or model.get("training_only") is not True
            or any(model.get(k) is not False for k in FLAGS)
            or type(model.get("seed")) is not int
            or not 0 <= model["seed"] < 2**32
            or type(model.get("std_raw")) not in (float, int)
            or not np.isfinite(model["std_raw"])
            or not 0.01 <= model["std_raw"] <= 0.15
        ):
            raise ValueError("sealed explicit output-memory sampling required")
        result = model["mean_model"]
    else:
        result = model
    validate_model(result)
    return cast(dict[str, Any], result)


def make_preview(model: dict[str, Any]) -> dict[str, Any]:
    current = mean_model(model)
    policy = make_policy(np.zeros((3, 12)), 0.25, model["model_hash"])
    policy["execution_profile"] = "causal_per_frame_neural_residual"
    policy["step_motor_proof"] = dict(
        schema="soccer.rsi.output_memory_step_preview.v1",
        model=model,
        decision_start_frame=30,
        nominal_target_is_pre_motor=True,
        force_input="previous_completed_frame",
        execution_source_hash=hash_bytes(Path(__file__).read_bytes()),
        qualification="UNQUALIFIED_SIM_TRAINING"
        if model.get("training_only")
        else "UNQUALIFIED_SIM_COUNTERFACTUAL",
        promotion_authorized=False,
    )
    policy["output_memory_motor_proof"] = dict(
        memory_hash=current["output_memory"]["memory_hash"],
        parent_model_hash=current["parent_model_hash"],
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledOutputMemoryMotor(CompiledKernelStepMotor):
    def __init__(self, policy: dict[str, Any]) -> None:
        wrapped = copy.deepcopy(policy["step_motor_proof"]["model"])
        if make_preview(wrapped) != policy:
            raise ValueError("output-memory execution commitment changed")
        model = mean_model(wrapped)
        super().__init__(parent_preview(model["frozen_parent"]))
        self._output_memory = validate_model(model)
        self._encoder_hash = self._output_memory.encoder_hash
        self._residual_layers = [
            (np.asarray(v["weight"]), np.asarray(v["bias"])) for v in model["residual_layers"]
        ]
        for weight, bias in self._residual_layers:
            weight.flags.writeable = False
            bias.flags.writeable = False
        self._sampling = wrapped if wrapped["schema"] == SAMPLING_SCHEMA else None
        self._policy_hash = policy["policy_hash"]

    def context(self, observation: Any, phase: int) -> np.ndarray[Any, Any]:
        if type(phase) is not int or not 0 <= phase <= 2:
            raise ValueError("causal physical phase required")
        return np.concatenate((self.features(observation)[:134], [float(phase)]))

    def raw_mean(self, observation: Any, phase: int) -> np.ndarray[Any, Any]:
        parent = super().raw_mean(observation, phase)
        context = self.context(observation, phase)
        hidden = context
        for weight, bias in self._residual_layers:
            hidden = np.tanh(weight @ hidden + bias)
        proposal = parent + 0.05 * hidden
        return np.asarray(
            self._output_memory.blend(context, proposal, encoder_hash=self._encoder_hash),
            dtype=np.float64,
        )
