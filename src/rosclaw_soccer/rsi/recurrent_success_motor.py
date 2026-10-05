"""Unqualified causal sequence student over an unchanged frozen proposal law."""

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.causal_residual_memory as memory_module
import rosclaw.growth.recurrent_residual_imitation as learner

from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.proposal_decoder_selection import select_proposal_decoder
from rosclaw_soccer.rsi.proposal_memory_motor import CompiledProposalMemoryMotor
from rosclaw_soccer.rsi.proposal_memory_motor import make_preview as parent_preview
from rosclaw_soccer.rsi.proposal_memory_motor import validate_model as validate_parent
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.recurrent_success_imitation_motor.v1"
FLAGS = (
    "promotion_authorized",
    "hardware_authorized",
    "runtime_execution_authorized",
    "fresh_holdout_open_authorized",
)
HIDDEN_DIMENSION = 64
STATE_FIELD = "recurrent_residual_hidden_state"


def validate_model(model: Any) -> None:
    if (
        type(model) is not dict
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
    ):
        raise ValueError("sealed non-authorizing causal sequence candidate required")
    parent = model.get("frozen_parent")
    if type(parent) is not dict:
        raise ValueError("complete frozen actual proposal parent required")
    validate_parent(parent)
    if model.get("parent_model_hash") != parent["model_hash"]:
        raise ValueError("exact frozen actual proposal identity required")
    if type(model.get("initial_seed")) is not int or not 0 <= model["initial_seed"] < 2**32:
        raise ValueError("bounded exact recurrent initialization seed required")
    parameters = model.get("parameters")
    memory = memory_module.CausalResidualMemory(parameters)
    if (memory.input_dimension, memory.output_dimension, memory.hidden_dimension) != (
        135,
        12,
        HIDDEN_DIMENSION,
    ):
        raise ValueError("canonical bounded recurrent dimensions required")
    receipt = model.get("learning_receipt")
    if receipt is None:
        if parameters != memory_module.initial_parameters(
            135, 12, hidden_dimension=HIDDEN_DIMENSION, seed=model["initial_seed"]
        ):
            raise ValueError("unreceipted recurrent parameter change rejected")
        return
    if type(receipt) is not dict or type(receipt.get("config")) is not dict:
        raise ValueError("complete causal sequence fit receipt required")
    if "executed_action_objective" in receipt:
        import rosclaw.growth.staged_action_projection as projection_module

        objective = receipt["executed_action_objective"]
        if (
            type(objective) is not dict
            or set(objective)
            != {
                "schema",
                "data_hash",
                "projection_source_hash",
                "cap",
                "slew",
                "raw_loss_weight",
                "order",
                "all_original_rows_validated",
                "teacher_actions_reconstructed",
                "teacher_forced_previous_actions_not_closed_loop_rollout",
                "projection_labels_are_recurrent_inputs",
                "physical_batch_verified",
                "promotion_authorized",
                "hardware_authorized",
            }
            or objective["schema"] != "rosclaw.growth.recurrent_executed_action_objective.v1"
            or type(objective["cap"]) is not float
            or objective["cap"] != 0.16
            or type(objective["slew"]) is not float
            or objective["slew"] != 0.012
            or type(objective["raw_loss_weight"]) is not float
            or not np.isfinite(objective["raw_loss_weight"])
            or not 0.001 <= objective["raw_loss_weight"] <= 1.0
            or objective["order"] != "CAP_TANH_THEN_SLEW_THEN_FINAL_BOX"
            or type(objective["all_original_rows_validated"]) is not int
            or objective["all_original_rows_validated"] != receipt.get("original_input_rows")
            or objective["projection_source_hash"]
            != hash_bytes(Path(projection_module.__file__).read_bytes())
            or type(objective["data_hash"]) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", objective["data_hash"]) is None
            or any(
                objective[k] is not True
                for k in (
                    "teacher_actions_reconstructed",
                    "teacher_forced_previous_actions_not_closed_loop_rollout",
                )
            )
            or any(
                objective[k] is not False
                for k in (
                    "projection_labels_are_recurrent_inputs",
                    "physical_batch_verified",
                    "promotion_authorized",
                    "hardware_authorized",
                )
            )
        ):
            raise ValueError("complete source-bound original executed-action objective required")
    config = receipt["config"]
    try:
        learner.RecurrentResidualImitationConfig(**config).validate()
    except (TypeError, ValueError) as error:
        raise ValueError("complete sequence configuration required") from error
    count, positive, history = (
        receipt.get("original_input_rows"),
        receipt.get("positive_weight_rows"),
        receipt.get("minibatch_loss_history"),
    )
    if (
        receipt.get("algorithm") != "DATASET_WEIGHTED_CAUSAL_RECURRENT_RESIDUAL_IMITATION_V1"
        or receipt.get("source_hash") != model["learner_source_hash"]
        or receipt.get("memory_source_hash") != model["memory_source_hash"]
        or receipt.get("fitted_parameters_hash") != hash_json(parameters)
        or receipt.get("behavior_model_hash") != parent["model_hash"]
        or receipt.get("activation_ceiling") != "SIM_ONLY"
        or config.get("residual_cap") != 0.2
        or config.get("learning_rate") != 0.0004
        or type(count) is not int
        or not 1080 <= count <= 199800
        or count % 270
        or type(positive) is not int
        or not 1080 <= positive <= count
        or positive % 270
        or type(receipt.get("original_episode_count")) is not int
        or receipt["original_episode_count"] * 270 != count
        or type(receipt.get("episode_horizon")) is not int
        or receipt["episode_horizon"] != 270
        or type(receipt.get("zero_weight_rows")) is not int
        or receipt["zero_weight_rows"] != count - positive
        or type(receipt.get("completed_optimizer_steps")) is not int
        or receipt["completed_optimizer_steps"] != config["steps"]
        or type(history) is not list
        or len(history) != config["steps"]
        or any(type(v) not in (float, int) or not np.isfinite(v) or v < 0 for v in history)
        or any(
            type(receipt.get(k)) not in (float, int)
            or not np.isfinite(receipt[k])
            or receipt[k] < 0
            for k in ("initial_full_positive_weight_loss", "final_full_positive_weight_loss")
        )
        or receipt["final_full_positive_weight_loss"]
        >= receipt["initial_full_positive_weight_loss"]
        or any(
            receipt.get(k) is not True
            for k in (
                "training_loss_improved",
                "state_reset_at_each_episode",
                "causal_architecture_not_input_feature_provenance",
                "zero_weight_rows_are_not_negative_gradient_examples",
                "teacher_selection_offline",
                "all_success_and_failure_episodes_retained",
            )
        )
        or any(
            receipt.get(k) is not False
            for k in (
                "teacher_targets_are_recurrent_inputs",
                "input_feature_causality_verified",
                "physical_batch_verified",
                "online_rl_claimed",
                "on_policy_ppo",
                "kl_trust_region_applied",
                "runtime_execution_authorized",
                "promotion_authorized",
                "hardware_authorized",
                "teacher_labels_are_actor_inputs",
                "private_fresh_accessed",
            )
        )
        or any(
            type(receipt.get(k)) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", receipt[k]) is None
            for k in ("physical_batch_hash", "teacher_selection_hash")
        )
    ):
        raise ValueError("complete non-authorizing causal sequence learning receipt required")


def make_model(
    parent: dict[str, Any],
    *,
    initial_seed: int = 0,
    parameters: Any = None,
    learning_receipt: Any = None,
) -> dict[str, Any]:
    validate_parent(parent)
    model = dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        frozen_parent=copy.deepcopy(parent),
        parent_model_hash=parent["model_hash"],
        initial_seed=initial_seed,
        parameters=copy.deepcopy(
            memory_module.initial_parameters(
                135, 12, hidden_dimension=HIDDEN_DIMENSION, seed=initial_seed
            )
            if parameters is None
            else parameters
        ),
        learning_receipt=copy.deepcopy(learning_receipt),
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
        schema="soccer.rsi.recurrent_success_imitation_preview.v1",
        model=model,
        decision_start_frame=30,
        nominal_target_is_pre_motor=True,
        force_input="previous_completed_frame",
        execution_source_hash=model["source_hash"],
        qualification="UNQUALIFIED_CAUSAL_SEQUENCE_IMITATION_NOT_PPO",
        promotion_authorized=False,
    )
    policy["recurrent_success_motor_proof"] = dict(
        parent_model_hash=model["parent_model_hash"],
        hidden_dimension=HIDDEN_DIMENSION,
        memory_zero_at_decision_start=True,
        hidden_state_logged_and_replayed_required=True,
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledRecurrentSuccessMotor(CompiledProposalMemoryMotor):
    def __init__(self, policy: dict[str, Any]) -> None:
        owned = copy.deepcopy(policy["step_motor_proof"]["model"])
        if make_preview(owned) != policy:
            raise ValueError("complete canonical sequence preview required")
        self._behavior = select_proposal_decoder(
            parent_preview(owned["frozen_parent"]), implementation="bounded_snapshot"
        )
        self._parent = self._behavior._parent
        self._guard = self._behavior._guard
        self._recurrent = memory_module.CausalResidualMemory(owned["parameters"])
        self._memory = ContactPhaseMemory()
        self._sampling = None
        self._policy_hash = policy["policy_hash"]
        self._active_frame: int | None = None

    @property
    def hidden_state(self) -> np.ndarray[Any, Any]:
        return np.array(self._recurrent.state, dtype=np.float64, copy=True)

    def features(self, observation: Any) -> Any:
        return self._behavior.features(observation)

    def raw_mean(self, observation: Any, phase: int) -> Any:
        if self._active_frame is None or self._active_frame < 30:
            raise ValueError(
                "recurrent state may advance only inside the sequential motor boundary"
            )
        baseline = self._behavior.raw_mean(observation, phase)
        context = np.concatenate((self.features(observation)[:134], [phase]))
        residual = self._recurrent.step(context, index=self._active_frame - 30)
        result = baseline + 0.2 * self._guard.gate(context) * residual
        if not np.isfinite(result).all():
            raise ValueError("nonfinite recurrent mean")
        return result

    def delta_at_frame(self, policy: dict[str, Any], body: Any, **boundary: Any) -> Any:
        if self._active_frame is not None:
            raise ValueError("reentrant recurrent boundary rejected")
        self._active_frame = boundary.get("frame")
        try:
            return super().delta_at_frame(policy, body, **boundary)
        finally:
            self._active_frame = None
