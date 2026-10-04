"""Separate SIM-only artifact for an explicitly longer frozen-batch optimizer.

The original parent and its receipt remain unchanged and fully validated.
Only its plastic head is replaced. Neither inner steps nor this adapter grant
fresh, promotion, runtime, or hardware authority.
"""

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.extended_proposal_regression as optimizer

from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.proposal_decoder_selection import select_proposal_decoder
from rosclaw_soccer.rsi.proposal_memory_motor import make_preview as parent_preview
from rosclaw_soccer.rsi.proposal_memory_motor import validate_model as validate_parent
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.extended_proposal_motor.v1"
FLAGS = (
    "promotion_authorized",
    "hardware_authorized",
    "runtime_execution_authorized",
    "fresh_holdout_open_authorized",
)


def validate_model(model: Any) -> None:
    if (
        type(model) is not dict
        or model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or any(model.get(k) is not False for k in FLAGS)
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or model.get("optimizer_source_hash") != hash_bytes(Path(optimizer.__file__).read_bytes())
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
        or type(model.get("maximum_inner_steps")) is not int
        or not 1 <= model["maximum_inner_steps"] <= 2560
        or model.get("physical_action_bounds_changed") is not False
        or model.get("protected_memory_changed") is not False
    ):
        raise ValueError("sealed non-authorizing extended proposal artifact required")
    parent = model.get("frozen_parent")
    if type(parent) is not dict:
        raise ValueError("complete immutable original parent required")
    validate_parent(parent)
    if model.get("parent_model_hash") != parent["model_hash"]:
        raise ValueError("exact frozen original parent identity required")
    layers = model.get("residual_layers")
    if type(layers) is not list or len(layers) != 3:
        raise ValueError("complete aligned extended plastic head required")
    for layer, old in zip(layers, parent["residual_layers"], strict=True):
        if type(layer) is not dict or set(layer) != {"weight", "bias"}:
            raise ValueError("complete aligned extended plastic head required")
        for name in ("weight", "bias"):
            value = np.asarray(layer[name])
            if (
                value.shape != np.asarray(old[name]).shape
                or value.dtype.kind not in "fiu"
                or not np.isfinite(value).all()
            ):
                raise ValueError("complete finite aligned extended plastic head required")
    receipt = model.get("learning_receipt")
    if receipt is None:
        if layers != parent["residual_layers"]:
            raise ValueError("initial copy cannot carry unreceipted plastic changes")
        return
    history = receipt.get("full_batch_loss_history") if type(receipt) is dict else None
    if (
        type(receipt) is not dict
        or receipt.get("algorithm") != "EXPLICIT_EXTENDED_PROPOSAL_REGRESSION_V1"
        or receipt.get("base_numeric_algorithm") != "PROPOSAL_TRUST_REGION_ADVANTAGE_REGRESSION_V1"
        or receipt.get("learner_parent_hash") != parent["model_hash"]
        or receipt.get("behavior_model_hash") != parent["model_hash"]
        or receipt.get("requested_inner_optimizer_steps") != model["maximum_inner_steps"]
        or type(receipt.get("requested_inner_optimizer_steps")) is not int
        or type(receipt.get("completed_optimizer_steps")) is not int
        or not 1 <= receipt["completed_optimizer_steps"] <= model["maximum_inner_steps"]
        or receipt.get("single_initial_behavior_reference") is not True
        or receipt.get("kl_budget_reset_between_passes") is not False
        or receipt.get("accepted_step_count_not_total_attempts") is not True
        or receipt.get("additional_physical_episodes_executed") != 0
        or type(receipt.get("additional_physical_episodes_executed")) is not int
        or receipt.get("online_rl_claimed") is not False
        or receipt.get("context_is_actor_observation") is not False
        or receipt.get("future_event_is_actor_observation") is not False
        or receipt.get("loss_weighting_profile") != parent["loss_weighting_profile"]
        or receipt.get("critic_profile") != parent.get("critic_profile", "whole-rollout")
        or type(receipt.get("physical_rollout_count")) is not int
        or not 4 <= receipt["physical_rollout_count"] <= 740
        or type(receipt.get("frame_sample_count")) is not int
        or receipt["frame_sample_count"] != 270 * receipt["physical_rollout_count"]
        or receipt.get("maximum_mean_kl") != parent["maximum_mean_kl"]
        or receipt.get("residual_cap") != 0.2
        or receipt.get("learning_rate") != parent["learning_rate"]
        or receipt.get("rho") != 0.9
        or receipt.get("frozen_baseline") is not True
        or receipt.get("frozen_guard") is not True
        or receipt.get("execution_ceiling") != "PROPOSAL_ONLY_NO_RUNTIME"
        or any(
            receipt.get(k) is not False
            for k in (
                "runtime_execution_authorized",
                "promotion_authorized",
                "hardware_authorized",
                "physical_batch_verified",
                "distributional_retention_guaranteed",
            )
        )
        or any(
            not isinstance(receipt.get(k), str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", receipt[k])
            for k in ("physical_batch_hash", "critic_evidence_hash", "offline_training_labels_hash")
        )
        or any(
            type(receipt.get(k)) not in (float, int)
            or not np.isfinite(receipt[k])
            or not 0 <= receipt[k] <= parent["maximum_mean_kl"]
            for k in ("exact_mean_conditional_kl", "exact_mean_marginal_kl")
        )
        or type(history) is not list
        or len(history) != receipt["completed_optimizer_steps"] + 1
        or any(type(v) not in (float, int) or not np.isfinite(v) for v in history)
        or any(a <= b for a, b in zip(history, history[1:], strict=False))
    ):
        raise ValueError("complete source-bound extended optimizer receipt required")
    likelihood = parent.get("optimizer_likelihood_profile", "marginal")
    if (likelihood == "marginal" and "likelihood_profile" in receipt) or (
        likelihood != "marginal" and receipt.get("likelihood_profile") != likelihood
    ):
        raise ValueError("same original likelihood objective required")
    device = parent.get("optimizer_compute_device", "cpu")
    if (device == "cpu" and "compute_device" in receipt) or (
        device != "cpu"
        and (
            receipt.get("compute_device") != device
            or receipt.get("cross_device_bit_identity_claimed") is not False
        )
    ):
        raise ValueError("same original explicit optimizer device required")


def make_model(
    parent: dict[str, Any],
    *,
    maximum_inner_steps: int,
    residual_layers: Any = None,
    learning_receipt: Any = None,
) -> dict[str, Any]:
    validate_parent(parent)
    model = dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        frozen_parent=copy.deepcopy(parent),
        parent_model_hash=parent["model_hash"],
        maximum_inner_steps=maximum_inner_steps,
        residual_layers=copy.deepcopy(
            parent["residual_layers"] if residual_layers is None else residual_layers
        ),
        learning_receipt=copy.deepcopy(learning_receipt),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        optimizer_source_hash=hash_bytes(Path(optimizer.__file__).read_bytes()),
        physical_action_bounds_changed=False,
        protected_memory_changed=False,
        **dict.fromkeys(FLAGS, False),
    )
    model["model_hash"] = hash_json(model)
    validate_model(model)
    return model


def make_preview(model: dict[str, Any]) -> dict[str, Any]:
    validate_model(model)
    policy: dict[str, Any] = make_policy(np.zeros((3, 12)), 0.25, model["model_hash"])
    policy["execution_profile"] = "causal_per_frame_neural_residual"
    policy["step_motor_proof"] = dict(
        schema="soccer.rsi.extended_proposal_preview.v1",
        model=model,
        decision_start_frame=30,
        nominal_target_is_pre_motor=True,
        force_input="previous_completed_frame",
        execution_source_hash=model["source_hash"],
        qualification="UNQUALIFIED_SIM_EXTENDED_OPTIMIZER",
        promotion_authorized=False,
    )
    policy["extended_proposal_motor_proof"] = dict(
        parent_model_hash=model["parent_model_hash"],
        maximum_inner_steps=model["maximum_inner_steps"],
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledExtendedProposalMotor:
    """Original bounded motor law and guards, separately receipted plastic head."""

    def __init__(self, policy: dict[str, Any]) -> None:
        model = copy.deepcopy(policy["step_motor_proof"]["model"])
        if make_preview(model) != policy:
            raise ValueError("complete extended preview binding required")
        parent = select_proposal_decoder(
            parent_preview(model["frozen_parent"]), implementation="bounded_snapshot"
        )
        self._decoder = parent
        self._parent, self._guard, self._cap = parent._parent, parent._guard, parent._cap
        self._layers = [
            (np.array(v["weight"], dtype=np.float64), np.array(v["bias"], dtype=np.float64))
            for v in model["residual_layers"]
        ]
        for w, b in self._layers:
            w.flags.writeable = b.flags.writeable = False
        self._zero = not np.any(self._layers[-1][0]) and not np.any(self._layers[-1][1])
        self._sampling, self._noise = None, None
        self._memory = ContactPhaseMemory()
        self._policy_hash = policy["policy_hash"]
        parent._layers, parent._zero = self._layers, self._zero
        parent._sampling, parent._noise = None, None
        parent._memory, parent._policy_hash = self._memory, self._policy_hash

    def features(self, observation: Any) -> Any:
        return self._decoder.features(observation)

    def raw_mean(self, observation: Any, phase: int) -> Any:
        return self._decoder.raw_mean(observation, phase)

    def delta_at_frame(self, policy: dict[str, Any], body: Any, **boundary: Any) -> Any:
        return self._decoder.delta_at_frame(policy, body, **boundary)
