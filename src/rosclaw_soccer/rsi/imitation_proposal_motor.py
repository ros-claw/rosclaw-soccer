"""Explicit supervised successful-teacher candidate, not PPO or authority.

Only an existing bounded plastic head is replaced. The complete prior,
protected state banks and original execution/slew/PD law remain unchanged.
Teacher success labels are offline labels, never runtime actor inputs.
"""

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.bounded_residual_imitation as learner

from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.proposal_decoder_selection import select_proposal_decoder
from rosclaw_soccer.rsi.proposal_memory_motor import make_preview as parent_preview
from rosclaw_soccer.rsi.proposal_memory_motor import validate_model as validate_parent
from rosclaw_soccer.rsi.step_motor_phase_context import ContactPhaseMemory
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.verified_success_imitation_motor.v1"
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
        or model.get("learner_source_hash") != hash_bytes(Path(learner.__file__).read_bytes())
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
        or model.get("physical_action_bounds_changed") is not False
        or model.get("protected_memory_changed") is not False
    ):
        raise ValueError("sealed non-authorizing bounded imitation candidate required")
    parent = model.get("frozen_parent")
    if type(parent) is not dict:
        raise ValueError("complete original frozen behavior required")
    validate_parent(parent)
    if model.get("parent_model_hash") != parent["model_hash"]:
        raise ValueError("exact original frozen behavior identity required")
    layers = model.get("residual_layers")
    if type(layers) is not list or len(layers) != 3:
        raise ValueError("complete bounded imitation layers required")
    for layer, original in zip(layers, parent["residual_layers"], strict=True):
        if type(layer) is not dict or set(layer) != {"weight", "bias"}:
            raise ValueError("complete aligned imitation layers required")
        for name in ("weight", "bias"):
            values = np.asarray(layer[name])
            if (
                values.dtype.kind not in "fiu"
                or values.shape != np.asarray(original[name]).shape
                or not np.isfinite(values).all()
                or np.max(np.abs(values.astype(np.float64))) > 1e6
            ):
                raise ValueError("finite aligned imitation layers required")
    receipt = model.get("learning_receipt")
    if receipt is None:
        if layers != parent["residual_layers"]:
            raise ValueError("unreceipted imitation changes rejected")
        return
    if type(receipt) is not dict or type(receipt.get("config")) is not dict:
        raise ValueError("complete bounded imitation fit receipt required")
    configuration = receipt["config"]
    try:
        learner.BoundedResidualImitationConfig(**configuration).validate()
    except (TypeError, ValueError) as error:
        raise ValueError("complete bounded imitation fit configuration required") from error
    count = receipt.get("original_input_rows")
    positive = receipt.get("positive_weight_rows")
    history = receipt.get("minibatch_loss_history")
    if (
        receipt.get("algorithm") != "DATASET_WEIGHTED_BOUNDED_RESIDUAL_IMITATION_V1"
        or receipt.get("source_hash") != model["learner_source_hash"]
        or receipt.get("fitted_layers_hash") != hash_json(layers)
        or receipt.get("activation_ceiling") != "SIM_ONLY"
        or configuration.get("residual_cap") != 0.2
        or configuration.get("learning_rate") != 0.0004
        or receipt.get("behavior_model_hash") != parent["model_hash"]
        or type(count) is not int
        or not 1080 <= count <= 200000
        or count % 270
        or type(positive) is not int
        or not 1080 <= positive <= count
        or positive % 270
        or type(receipt.get("zero_weight_rows")) is not int
        or receipt["zero_weight_rows"] != count - positive
        or receipt.get("completed_optimizer_steps") != configuration["steps"]
        or type(receipt.get("completed_optimizer_steps")) is not int
        or type(history) is not list
        or len(history) != configuration["steps"]
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
                "teacher_rows_are_not_new_physical_episodes",
                "zero_weight_rows_are_not_negative_gradient_examples",
                "teacher_selection_offline",
                "all_success_and_failure_episodes_retained",
            )
        )
        or any(
            receipt.get(k) is not False
            for k in (
                "physical_batch_verified",
                "temporal_teacher_causality_verified",
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
        raise ValueError("complete non-authorizing supervised imitation receipt required")


def make_model(
    parent: dict[str, Any], *, residual_layers: Any = None, learning_receipt: Any = None
) -> dict[str, Any]:
    validate_parent(parent)
    model: dict[str, Any] = dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        frozen_parent=copy.deepcopy(parent),
        parent_model_hash=parent["model_hash"],
        residual_layers=copy.deepcopy(
            parent["residual_layers"] if residual_layers is None else residual_layers
        ),
        learning_receipt=copy.deepcopy(learning_receipt),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
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
    policy: dict[str, Any] = make_policy(np.zeros((3, 12)), 0.25, model["model_hash"])
    policy["execution_profile"] = "causal_per_frame_neural_residual"
    policy["step_motor_proof"] = dict(
        schema="soccer.rsi.verified_success_imitation_preview.v1",
        model=model,
        decision_start_frame=30,
        nominal_target_is_pre_motor=True,
        force_input="previous_completed_frame",
        execution_source_hash=model["source_hash"],
        qualification="UNQUALIFIED_SIM_SUPERVISED_IMITATION_NOT_PPO",
        promotion_authorized=False,
    )
    policy["verified_success_imitation_motor_proof"] = dict(
        parent_model_hash=model["parent_model_hash"],
        offline_teacher_only=True,
        original_execution_caps_retained=True,
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledImitationProposalMotor:
    def __init__(self, policy: dict[str, Any]) -> None:
        model = copy.deepcopy(policy["step_motor_proof"]["model"])
        if make_preview(model) != policy:
            raise ValueError("complete supervised imitation preview required")
        decoder = select_proposal_decoder(
            parent_preview(model["frozen_parent"]), implementation="bounded_snapshot"
        )
        self._layers = [
            (np.array(v["weight"], dtype=np.float64), np.array(v["bias"], dtype=np.float64))
            for v in model["residual_layers"]
        ]
        for w, b in self._layers:
            w.flags.writeable = b.flags.writeable = False
        self._decoder = decoder
        self._parent, self._guard, self._cap = decoder._parent, decoder._guard, decoder._cap
        self._memory = ContactPhaseMemory()
        self._zero = not np.any(self._layers[-1][0]) and not np.any(self._layers[-1][1])
        self._policy_hash = policy["policy_hash"]
        decoder._layers, decoder._zero = self._layers, self._zero
        decoder._memory, decoder._policy_hash = self._memory, self._policy_hash
        decoder._sampling, decoder._noise = None, None

    def features(self, observation: Any) -> Any:
        return self._decoder.features(observation)

    def raw_mean(self, observation: Any, phase: int) -> Any:
        return self._decoder.raw_mean(observation, phase)

    def delta_at_frame(self, policy: dict[str, Any], body: Any, **boundary: Any) -> Any:
        return self._decoder.delta_at_frame(policy, body, **boundary)
