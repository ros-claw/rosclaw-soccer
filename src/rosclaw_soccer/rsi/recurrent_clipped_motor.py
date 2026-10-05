"""Flat SIM-only recurrent policy-gradient state over a frozen proposal law.

Numeric fit receipts bind current and previous parameters, not physical data
provenance. The MC value network is retained but never used as a motor input.
"""

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.causal_residual_memory as memory_module
import rosclaw.growth.recurrent_clipped_actor_critic as learner

from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.proposal_decoder_selection import select_proposal_decoder
from rosclaw_soccer.rsi.proposal_memory_motor import make_preview as parent_preview
from rosclaw_soccer.rsi.proposal_memory_motor import validate_model as validate_parent
from rosclaw_soccer.rsi.recurrent_success_motor import FLAGS, CompiledRecurrentSuccessMotor
from rosclaw_soccer.rsi.recurrent_success_motor import SCHEMA as IMITATION_SCHEMA
from rosclaw_soccer.rsi.recurrent_success_motor import validate_model as validate_imitation
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.recurrent_clipped_mc_motor.v1"


def _digest(value: Any) -> bool:
    return type(value) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _finite(value: Any, *, low: float = 0.0, high: float = 1e6) -> bool:
    return type(value) in (int, float) and np.isfinite(value) and low <= value <= high


def _memory(parameters: Any) -> None:
    memory = memory_module.CausalResidualMemory(parameters)
    if (memory.input_dimension, memory.output_dimension, memory.hidden_dimension) != (135, 12, 64):
        raise ValueError("canonical recurrent motor dimensions required")


def validate_model(model: Any) -> None:
    expected = {
        "schema",
        "activation_ceiling",
        "frozen_parent",
        "parent_model_hash",
        "previous_model_hash",
        "previous_actor_parameters",
        "previous_critic_parameters",
        "parameters",
        "critic_parameters",
        "learning_receipt",
        "source_hash",
        "memory_source_hash",
        "learner_source_hash",
        "physical_action_bounds_changed",
        "protected_memory_changed",
        "model_hash",
        *FLAGS,
    }
    if (
        type(model) is not dict
        or set(model) != expected
        or model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or any(model.get(k) is not False for k in FLAGS)
        or model.get("physical_action_bounds_changed") is not False
        or model.get("protected_memory_changed") is not False
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or model.get("memory_source_hash") != hash_bytes(Path(memory_module.__file__).read_bytes())
        or model.get("learner_source_hash") != hash_bytes(Path(learner.__file__).read_bytes())
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
        or not _digest(model.get("previous_model_hash"))
    ):
        raise ValueError("sealed non-authorizing flat recurrent policy-gradient model required")
    validate_parent(model["frozen_parent"])
    if model["parent_model_hash"] != model["frozen_parent"]["model_hash"]:
        raise ValueError("unchanged actual proposal parent required")
    for name in ("parameters", "previous_actor_parameters"):
        _memory(model[name])
    for name in ("critic_parameters", "previous_critic_parameters"):
        learner._value_parameters(model[name], 135)
    receipt = model["learning_receipt"]
    if type(receipt) is not dict or type(receipt.get("config")) is not dict:
        raise ValueError("complete recurrent clipped fit receipt required")
    try:
        config = learner.RecurrentClippedActorCriticConfig(**receipt["config"])
        config.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("complete recurrent clipped configuration required") from error
    count, accepted, rejected = (
        receipt.get("episode_count"),
        receipt.get("accepted_joint_optimizer_steps"),
        receipt.get("rejected_and_rolled_back_updates"),
    )
    history = receipt.get("accepted_update_history")
    if (
        receipt.get("algorithm") != "RECURRENT_CLIPPED_LATENT_POLICY_WITH_MC_VALUE_FITTING_V1"
        or receipt.get("source_hash") != model["learner_source_hash"]
        or receipt.get("memory_source_hash") != model["memory_source_hash"]
        or receipt.get("original_actor_parameters_hash")
        != hash_json(model["previous_actor_parameters"])
        or receipt.get("fitted_actor_parameters_hash") != hash_json(model["parameters"])
        or receipt.get("original_critic_parameters_hash")
        != hash_json(model["previous_critic_parameters"])
        or receipt.get("fitted_critic_parameters_hash") != hash_json(model["critic_parameters"])
        or receipt.get("actor_parameters_changed")
        is not (hash_json(model["previous_actor_parameters"]) != hash_json(model["parameters"]))
        or not _digest(receipt.get("input_numeric_hash"))
        or (config.residual_cap, config.std, config.rho) != (0.2, 0.1, 0.9)
        or type(count) is not int
        or not 4 <= count <= 740
        or type(receipt.get("episode_horizon")) is not int
        or receipt["episode_horizon"] != 270
        or type(accepted) is not int
        or not 0 <= accepted <= config.steps
        or type(rejected) is not int
        or rejected not in (0, 1)
        or (accepted != config.steps if rejected == 0 else accepted >= config.steps)
        or type(history) is not list
        or len(history) != accepted
        or any(
            type(row) is not dict
            or set(row) != {"actor_loss", "value_loss", "marginal_mean_kl", "conditional_mean_kl"}
            or not _finite(row["actor_loss"], low=-1e6)
            or not _finite(row["value_loss"])
            or any(
                not _finite(row[k], high=config.maximum_mean_kl)
                for k in ("marginal_mean_kl", "conditional_mean_kl")
            )
            for row in history
        )
        or any(
            not _finite(receipt.get(k), high=config.maximum_mean_kl)
            for k in ("full_batch_marginal_mean_kl", "full_batch_conditional_mean_kl")
        )
        or not _finite(receipt.get("behavior_density_max_abs_error"), high=1e-8)
        or not _finite(receipt.get("final_mc_value_mse"))
        or any(
            type(receipt.get(k)) is not int or not 0 <= receipt[k] <= count * 270
            for k in ("positive_advantage_rows", "negative_advantage_rows", "zero_advantage_rows")
        )
        or sum(
            receipt[k]
            for k in ("positive_advantage_rows", "negative_advantage_rows", "zero_advantage_rows")
        )
        != count * 270
        or any(
            receipt.get(k) is not True
            for k in (
                "all_rows_behavior_density_validated",
                "all_rows_in_kl_checks",
                "state_reset_at_each_episode",
                "previous_candidate_mean_in_ar_conditioning",
                "latent_likelihood_not_projected_action_likelihood",
            )
        )
        or any(
            receipt.get(k) is not False
            for k in (
                "frozen_advantages_are_actor_inputs",
                "returns_are_actor_inputs",
                "td_bootstrapping",
                "on_policy_collection_provenance_verified",
                "physical_batch_verified",
                "promotion_authorized",
                "runtime_execution_authorized",
                "hardware_authorized",
            )
        )
        or "actor_parameters" in receipt
        or "critic_parameters" in receipt
    ):
        raise ValueError("complete bounded original-density actor/MC-critic fit receipt required")


def make_model(
    origin: dict[str, Any], fit: dict[str, Any], *, initial_critic: Any = None
) -> dict[str, Any]:
    if type(origin) is not dict:
        raise ValueError("complete actual recurrent origin required")
    if origin.get("schema") == IMITATION_SCHEMA:
        validate_imitation(origin)
        if initial_critic is None:
            raise ValueError("explicit original MC critic state required on first update")
        previous_critic = initial_critic
    elif origin.get("schema") == SCHEMA:
        validate_model(origin)
        if initial_critic is not None:
            raise ValueError("continuation must preserve its actual previous critic")
        previous_critic = origin["critic_parameters"]
    else:
        raise ValueError("allowlisted actual recurrent origin required")
    if type(fit) is not dict or not {"actor_parameters", "critic_parameters"} <= set(fit):
        raise ValueError("complete actual actor and critic fit required")
    model = dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        frozen_parent=copy.deepcopy(origin["frozen_parent"]),
        parent_model_hash=origin["frozen_parent"]["model_hash"],
        previous_model_hash=origin["model_hash"],
        previous_actor_parameters=copy.deepcopy(origin["parameters"]),
        previous_critic_parameters=copy.deepcopy(previous_critic),
        parameters=copy.deepcopy(fit["actor_parameters"]),
        critic_parameters=copy.deepcopy(fit["critic_parameters"]),
        learning_receipt=copy.deepcopy(
            {k: v for k, v in fit.items() if k not in ("actor_parameters", "critic_parameters")}
        ),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        memory_source_hash=hash_bytes(Path(memory_module.__file__).read_bytes()),
        learner_source_hash=hash_bytes(Path(learner.__file__).read_bytes()),
        physical_action_bounds_changed=False,
        protected_memory_changed=False,
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
        schema="soccer.rsi.recurrent_clipped_mc_preview.v1",
        model=copy.deepcopy(model),
        decision_start_frame=30,
        nominal_target_is_pre_motor=True,
        force_input="previous_completed_frame",
        execution_source_hash=model["source_hash"],
        qualification="UNQUALIFIED_OFFLINE_CLIPPED_MC_NOT_ONLINE_RL",
        promotion_authorized=False,
    )
    policy["recurrent_clipped_motor_proof"] = dict(
        parent_model_hash=model["parent_model_hash"],
        previous_model_hash=model["previous_model_hash"],
        hidden_dimension=64,
        memory_zero_at_decision_start=True,
        hidden_state_logged_and_replayed_required=True,
        critic_is_motor_input=False,
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledRecurrentClippedMotor(CompiledRecurrentSuccessMotor):
    def __init__(self, policy: dict[str, Any]) -> None:
        owned = copy.deepcopy(policy.get("step_motor_proof", {}).get("model", {}))
        if hash_json(make_preview(owned)) != hash_json(policy):
            raise ValueError("complete canonical clipped sequence preview required")
        self._behavior = select_proposal_decoder(
            parent_preview(owned["frozen_parent"]), implementation="bounded_snapshot"
        )
        self._parent, self._guard = self._behavior._parent, self._behavior._guard
        self._recurrent = memory_module.CausalResidualMemory(owned["parameters"])
        self._memory = ContactPhaseMemory()
        self._sampling = None
        self._policy_hash = policy["policy_hash"]
        self._active_frame: int | None = None
