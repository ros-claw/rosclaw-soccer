"""AR(1) exploration around an independently identified frozen memory parent.

The stationary marginal scale is unchanged; conditional innovation scales are
different. Training must reconstruct candidate previous means, not subtract a
fixed old noise offset. This is simulation training, never execution authority.
"""

import copy
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import rosclaw.growth.correlated_exploration as exploration
from rosclaw.growth.anchor_kernel import AnchorKernelGuard
from rosclaw.growth.correlated_exploration import conditional_scale, stationary_noise

from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.kernel_guarded_step_execution import CompiledKernelStepMotor
from rosclaw_soccer.rsi.kernel_guarded_step_network import FLAGS
from rosclaw_soccer.rsi.output_memory_step_motor import CompiledOutputMemoryMotor
from rosclaw_soccer.rsi.output_memory_step_motor import make_preview as parent_preview
from rosclaw_soccer.rsi.output_memory_step_motor import validate_model as parent_validate
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.smooth_memory_motor.v1"
SAMPLING_SCHEMA = "soccer.rsi.smooth_memory_sampling.v1"


def validate_model(model: dict[str, Any]) -> None:
    if (
        model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or model.get("core_exploration_source_hash")
        != hash_bytes(Path(exploration.__file__).read_bytes())
        or any(model.get(k) is not False for k in FLAGS)
        or type(model.get("generation")) is not int
        or not 0 <= model["generation"] <= 32
        or model.get("raw_residual_cap") != 0.05
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
    ):
        raise ValueError("sealed SIM-only smooth-memory actor required")
    parent_validate(model["frozen_parent"])
    if (
        model["parent_model_hash"] != model["frozen_parent"]["model_hash"]
        or len(model["residual_layers"]) != 3
    ):
        raise ValueError("complete immutable memory parent and plastic layers required")
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
            raise ValueError("finite aligned plastic smooth-memory network required")
    receipt = model["learning_receipt"]
    if model["generation"] == 0:
        last = model["residual_layers"][-1]
        if receipt is not None or np.any(last["weight"]) or np.any(last["bias"]):
            raise ValueError("zero residual initialization required")
    elif (
        not isinstance(receipt, dict)
        or receipt.get("algorithm") != "SMOOTH_MEMORY_AR1_PPO_MC_TERMINAL"
        or any(
            not re.fullmatch(r"sha256:[0-9a-f]{64}", receipt.get(k, ""))
            for k in ("physical_batch_hash", "learner_parent_hash", "optimizer_source_hash")
        )
        or type(receipt.get("completed_optimizer_steps")) is not int
        or not 1 <= receipt["completed_optimizer_steps"] <= 160
        or receipt.get("all_residual_layers_trainable") is not True
        or receipt.get("frozen_parent") is not True
        or receipt.get("distributional_retention_guaranteed") is not False
        or any(
            type(receipt.get(k)) not in (float, int)
            or not np.isfinite(receipt[k])
            or not 0 <= receipt[k] <= 0.005
            for k in ("exact_mean_conditional_kl", "exact_mean_marginal_kl")
        )
        or any(receipt.get(k) is not False for k in ("promotion_authorized", "hardware_authorized"))
    ):
        raise ValueError("actual conditional-density learning receipt required")


def initial_model(parent: dict[str, Any]) -> dict[str, Any]:
    parent_validate(parent)
    layers = copy.deepcopy(parent["residual_layers"])
    layers[-1] = dict(weight=np.zeros((12, 128)).tolist(), bias=np.zeros(12).tolist())
    model = dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        generation=0,
        parent_model_hash=parent["model_hash"],
        frozen_parent=copy.deepcopy(parent),
        residual_layers=layers,
        raw_residual_cap=0.05,
        learning_receipt=None,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        core_exploration_source_hash=hash_bytes(Path(exploration.__file__).read_bytes()),
        **dict.fromkeys(FLAGS, False),
    )
    model["model_hash"] = hash_json(model)
    validate_model(model)
    return model


def make_sampling_view(
    model: dict[str, Any], *, seed: int, std: float, rho: float = 0.9
) -> dict[str, Any]:
    validate_model(model)
    if (
        type(seed) is not int
        or not 0 <= seed < 2**32
        or type(std) not in (float, int)
        or not np.isfinite(std)
        or not 0.01 <= std <= 0.15
        or type(rho) not in (float, int)
        or not np.isfinite(rho)
        or not 0 <= rho <= 0.95
        or conditional_scale(std, rho, first=False) < 0.01
    ):
        raise ValueError("bounded stationary exploration required")
    view = dict(
        schema=SAMPLING_SCHEMA,
        mean_model=copy.deepcopy(model),
        seed=seed,
        std_raw=float(std),
        rho=float(rho),
        activation_ceiling="SIM_ONLY",
        training_only=True,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        **dict.fromkeys(FLAGS, False),
    )
    view["model_hash"] = hash_json(view)
    return view


def mean_model(wrapped: dict[str, Any]) -> dict[str, Any]:
    if wrapped.get("schema") == SAMPLING_SCHEMA:
        model = wrapped["mean_model"]
        validate_model(model)
        if (
            wrapped.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
            or wrapped.get("model_hash")
            != hash_json({k: v for k, v in wrapped.items() if k != "model_hash"})
            or wrapped.get("activation_ceiling") != "SIM_ONLY"
            or wrapped.get("training_only") is not True
            or any(wrapped.get(k) is not False for k in FLAGS)
            or type(wrapped.get("seed")) is not int
            or not 0 <= wrapped["seed"] < 2**32
            or type(wrapped.get("std_raw")) not in (float, int)
            or not np.isfinite(wrapped["std_raw"])
            or not 0.01 <= wrapped["std_raw"] <= 0.15
            or type(wrapped.get("rho")) not in (float, int)
            or not np.isfinite(wrapped["rho"])
            or not 0 <= wrapped["rho"] <= 0.95
            or conditional_scale(wrapped["std_raw"], wrapped["rho"], first=False) < 0.01
        ):
            raise ValueError("stationary exploration commitment changed")
    else:
        model = wrapped
        validate_model(model)
    return cast(dict[str, Any], model)


def make_preview(wrapped: dict[str, Any]) -> dict[str, Any]:
    model = mean_model(wrapped)
    policy = make_policy(np.zeros((3, 12)), 0.25, wrapped["model_hash"])
    policy["execution_profile"] = "causal_per_frame_neural_residual"
    policy["step_motor_proof"] = dict(
        schema="soccer.rsi.smooth_memory_preview.v1",
        model=wrapped,
        decision_start_frame=30,
        nominal_target_is_pre_motor=True,
        force_input="previous_completed_frame",
        execution_source_hash=hash_bytes(Path(__file__).read_bytes()),
        qualification="UNQUALIFIED_SIM_TRAINING"
        if wrapped.get("training_only")
        else "UNQUALIFIED_SIM_COUNTERFACTUAL",
        promotion_authorized=False,
    )
    policy["smooth_memory_motor_proof"] = dict(
        parent_model_hash=model["parent_model_hash"],
        memory_hash=model["frozen_parent"]["output_memory"]["memory_hash"],
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledSmoothMemoryMotor(CompiledKernelStepMotor):
    def __init__(self, policy: dict[str, Any]) -> None:
        wrapped = copy.deepcopy(policy["step_motor_proof"]["model"])
        if make_preview(wrapped) != policy:
            raise ValueError("smooth-memory execution commitment changed")
        model = mean_model(wrapped)
        self._parent = CompiledOutputMemoryMotor(parent_preview(model["frozen_parent"]))
        self._guard = AnchorKernelGuard(
            model["frozen_parent"]["output_memory"]["observations"], bandwidth=1e-4
        )
        self._layers = [
            (np.asarray(v["weight"], dtype=np.float64), np.asarray(v["bias"], dtype=np.float64))
            for v in model["residual_layers"]
        ]
        for weight, bias in self._layers:
            weight.flags.writeable = bias.flags.writeable = False
        self._zero = not np.any(self._layers[-1][0]) and not np.any(self._layers[-1][1])
        self._sampling = wrapped if wrapped["schema"] == SAMPLING_SCHEMA else None
        self._noise: Any = (
            stationary_noise(
                seed=wrapped["seed"], rho=wrapped["rho"], count=270, dimension=12, first_frame=30
            )
            if self._sampling is not None
            else None
        )
        self._memory = ContactPhaseMemory()
        self._policy_hash = policy["policy_hash"]

    def features(self, observation: Any) -> Any:
        return self._parent.features(observation)

    def raw_mean(self, observation: Any, phase: int) -> Any:
        mean = self._parent.raw_mean(observation, phase)
        if self._zero:
            return mean
        context = np.concatenate((self.features(observation)[:134], [phase]))
        gate = self._guard.gate(context)
        if gate == 0:
            return mean
        hidden = context
        for weight, bias in self._layers:
            hidden = np.tanh(weight @ hidden + bias)
        result = mean + 0.05 * gate * hidden
        if not np.isfinite(result).all():
            raise ValueError("nonfinite smooth-memory actor mean")
        return result

    def latent_sample(self, observation: Any, frame: int, phase: int) -> tuple[Any, float]:
        if self._sampling is None or type(frame) is not int or not 30 <= frame < 300:
            raise ValueError("declared stationary exploration frame required")
        view = self._sampling
        mean = self.raw_mean(observation, phase)
        index = frame - 30
        noise = self._noise[index]
        offset = np.zeros(12) if index == 0 else view["rho"] * self._noise[index - 1]
        scale = conditional_scale(view["std_raw"], view["rho"], first=index == 0)
        innovation = view["std_raw"] * (noise - offset)
        logp = np.sum(-0.5 * (innovation / scale) ** 2 - np.log(scale) - 0.5 * np.log(2 * np.pi))
        return mean + view["std_raw"] * noise, float(logp)
