"""A separate SIM-only AWR-inspired actor family over the current NN memory.

Historical PPO/model modules remain artifact-bound and unchanged. This adapter
outputs bounded joint-position residuals, never direct torque or authorization.
"""

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.domain_anchor_bank as domain_module
import rosclaw.growth.event_credit_partitions as event_module
import rosclaw.growth.indexed_anchor_output_memory as index_module
import rosclaw.growth.proposal_advantage_regression as regression_module
import rosclaw.growth.sample_weighting as weighting_module
from rosclaw.growth.domain_anchor_bank import DomainAnchorGuard
from rosclaw.growth.indexed_anchor_output_memory import IndexedAnchorOutputMemory

from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.current_memory_motor import CompiledCurrentMemoryMotor
from rosclaw_soccer.rsi.current_memory_motor import make_preview as initial_preview
from rosclaw_soccer.rsi.current_memory_motor import validate_model as validate_initial
from rosclaw_soccer.rsi.domain_memory_protection import protection_identity, validate_protection
from rosclaw_soccer.rsi.kernel_guarded_step_network import FLAGS
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.proposal_memory_motor.v1"
LOSS_PROFILES = ("uniform-frame", "equal-contact-phase-mass", "equal-first-contact-lead-mass")
CRITIC_PROFILES = ("whole-rollout", "whole-context", "whole-context-neural")


def critic_kind(profile: str) -> str:
    return {
        "whole-rollout": "WHOLE_TRAJECTORY_CROSSFIT_MC_NOT_TD_LAMBDA",
        "whole-context": "WHOLE_CONTEXT_CROSSFIT_MC_RAW_RETURN_NOT_TD_LAMBDA",
        "whole-context-neural": "WHOLE_CONTEXT_NEURAL_CROSSFIT_MC_NOT_TD_OR_GAE",
    }[profile]


def _validate_neural_critic(model: dict[str, Any]) -> None:
    from rosclaw.growth import context_prediction_mlp, neural_context_advantages

    if model.get("core_context_source_hash") != hash_bytes(
        Path(neural_context_advantages.__file__).read_bytes()
    ) or model.get("core_prediction_source_hash") != hash_bytes(
        Path(context_prediction_mlp.__file__).read_bytes()
    ):
        raise ValueError("exact neural MC prediction and credit source commitments required")
    fits = model.get("neural_critic_fit_results")
    if model["generation"] == 0:
        if fits is not None:
            raise ValueError("untrained proposal cannot carry fitted neural critics")
        return
    if type(fits) is not list or len(fits) != 4:
        raise ValueError("four complete source-bound neural critics required")
    if type(model.get("learning_receipt")) is not dict:
        raise ValueError("complete neural MC learning receipt required")
    if (
        type(model["learning_receipt"].get("frame_sample_count")) is not int
        or not 1080 <= model["learning_receipt"]["frame_sample_count"] <= 200000
    ):
        raise ValueError("bounded complete neural MC physical frame count required")
    union = set()
    for fit in fits:
        if type(fit) is not dict or type(fit.get("held_out_contexts")) is not list:
            raise ValueError("complete neural critic context identities required")
        if (
            not fit["held_out_contexts"]
            or any(type(v) is not int or v < 0 for v in fit["held_out_contexts"])
            or len(set(fit["held_out_contexts"])) != len(fit["held_out_contexts"])
        ):
            raise ValueError("unique integer held-out physical context identities required")
        union.update(fit["held_out_contexts"])
    labels = sorted(union)
    if len(labels) != model["learning_receipt"].get("independent_critic_contexts"):
        raise ValueError("exact neural critic independent context count required")
    hashes = []
    for fold, fit in enumerate(fits):
        if (
            fit.get("held_out_contexts") != labels[fold::4]
            or fit.get("train_contexts") != [v for v in labels if v not in labels[fold::4]]
            or fit.get("algorithm") != "CONTEXT_DISJOINT_SUPERVISED_MLP_NOT_RL"
            or type(fit.get("optimizer_updates")) is not int
            or not 1 <= fit["optimizer_updates"] <= 25000000
            or type(fit.get("train_contexts")) is not list
            or any(type(v) is not int for v in fit["train_contexts"])
            or type(fit.get("train_rows")) is not int
            or type(fit.get("held_out_rows")) is not int
            or fit["train_rows"] < 32
            or fit["held_out_rows"] < 16
            or fit["train_rows"] + fit["held_out_rows"]
            != model["learning_receipt"]["frame_sample_count"]
            or fit.get("input_normalization_training_contexts_only") is not True
            or fit.get("hyperparameters_chosen_on_holdout") is not False
            or type(fit.get("motor_policy_updates")) is not int
            or fit["motor_policy_updates"] != 0
            or any(fit.get(k) is not False for k in ("promotion_authorized", "hardware_authorized"))
        ):
            raise ValueError("complete disjoint non-authorizing neural critic fit required")
        network = fit.get("model")
        if type(network) is not dict:
            raise ValueError("complete neural MC predictor required")
        mean = np.asarray(network.get("input_mean"))
        if mean.shape != (513,) or context_prediction_mlp.predict(network, mean[None]).shape != (
            1,
            1,
        ):
            raise ValueError("exact 512 state features plus phase scalar critic required")
        hashes.append(network["model_hash"])
    receipt = model["learning_receipt"]
    if (
        receipt.get("critic_model_hashes") != hashes
        or receipt.get("core_prediction_source_hash") != model["core_prediction_source_hash"]
        or receipt.get("critic_readout_role") != "LINEAR_DIAGNOSTIC_ONLY_NEURAL_USED_FOR_ADVANTAGES"
    ):
        raise ValueError("complete neural MC critic identity receipt required")


def validate_model(model: dict[str, Any]) -> None:
    if (
        model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or model.get("core_regression_source_hash")
        != hash_bytes(Path(regression_module.__file__).read_bytes())
        or model.get("core_index_source_hash")
        != hash_bytes(Path(index_module.__file__).read_bytes())
        or model.get("core_weighting_source_hash")
        != hash_bytes(Path(weighting_module.__file__).read_bytes())
        or model.get("loss_weighting_profile") not in LOSS_PROFILES
        or model.get("critic_profile", "whole-rollout") not in CRITIC_PROFILES
        or model.get("core_domain_source_hash")
        != hash_bytes(Path(domain_module.__file__).read_bytes())
        or model.get("core_event_source_hash")
        != hash_bytes(Path(event_module.__file__).read_bytes())
        or "protected_domain_bank" not in model
        or model.get("memory_query_representation") != "EXACT_COORDINATE_INDEX_INTACT_LOGICAL_BANK"
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
        raise ValueError("sealed bounded SIM-only proposal-memory actor required")
    regression_module.ProposalAdvantageRegressionConfig(
        maximum_mean_kl=model.get("maximum_mean_kl"),
        execution_ceiling=model.get("proposal_execution_ceiling"),
    ).validate()
    if model.get("runtime_execution_authorized") is not False:
        raise ValueError("proposal cannot grant runtime authority")
    if model.get("critic_profile", "whole-rollout") == "whole-context":
        from rosclaw.growth import context_crossfit as context_module

        if model.get("core_context_source_hash") != hash_bytes(
            Path(context_module.__file__).read_bytes()
        ):
            raise ValueError("exact context critic source commitment required")
    if model.get("critic_profile") == "whole-context-neural":
        _validate_neural_critic(model)
    initial = model["initial_actor"]
    validate_initial(initial)
    if model["protected_domain_bank"] is not None:
        validate_protection(model["protected_domain_bank"], initial)
    protected_hash, protected_rows, protected_contexts = protection_identity(model)
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
    _validate_weighting_receipt(model, receipt)
    if (
        not isinstance(receipt, dict)
        or receipt.get("algorithm") != "PROPOSAL_TRUST_REGION_ADVANTAGE_REGRESSION_V1"
        or receipt.get("residual_cap") != 0.2
        or receipt.get("learning_rate") != 0.0004
        or receipt.get("maximum_mean_kl") != model["maximum_mean_kl"]
        or receipt.get("execution_ceiling") != "PROPOSAL_ONLY_NO_RUNTIME"
        or receipt.get("runtime_execution_authorized") is not False
        or receipt.get("rho") != 0.9
        or receipt.get("temperature") != 0.5
        or receipt.get("maximum_weight") != 20.0
        or receipt.get("critic_kind") != critic_kind(model.get("critic_profile", "whole-rollout"))
        or receipt.get("critic_crossfit_folds") != 4
        or type(receipt.get("physical_rollout_count")) is not int
        or not 4 <= receipt["physical_rollout_count"] <= 740
        or type(receipt.get("frame_sample_count")) is not int
        or receipt["frame_sample_count"] != receipt["physical_rollout_count"] * 270
        or receipt.get("optimizer_source_hash") != model["core_regression_source_hash"]
        or receipt.get("adapter_source_hash")
        != hash_bytes(Path(__file__).with_name("proposal_memory_learning.py").read_bytes())
        or receipt.get("behavior_model_hash") != initial["baseline"]["base_model"]["model_hash"]
        or receipt.get("protected_memory_hash") != protected_hash
        or receipt.get("protected_memory_rows") != protected_rows
        or receipt.get("protected_anchor_contexts") != protected_contexts
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
            or not 0 <= receipt[k] <= model["maximum_mean_kl"]
            for k in ("exact_mean_conditional_kl", "exact_mean_marginal_kl")
        )
    ):
        raise ValueError("complete bounded advantage regression receipt required")
    if receipt["learner_parent_hash"] != hash_json(
        _initial_descriptor(
            initial,
            maximum_mean_kl=model["maximum_mean_kl"],
            loss_weighting_profile=model["loss_weighting_profile"],
            protected_domain_bank=model["protected_domain_bank"],
            critic_profile=model.get("critic_profile", "whole-rollout"),
        )
    ):
        raise ValueError("exact zero-addition learner parent receipt required")
    if model.get("critic_profile", "whole-rollout") in (
        "whole-context",
        "whole-context-neural",
    ) and (
        receipt.get("critic_readout_unit") != "raw_terminal_return"
        or receipt.get("context_is_actor_observation") is not False
        or receipt.get("core_context_source_hash") != model["core_context_source_hash"]
        or type(receipt.get("independent_critic_contexts")) is not int
        or not 4 <= receipt["independent_critic_contexts"] <= receipt["physical_rollout_count"]
        or receipt.get("overlapping_context_count") != 0
        or type(receipt.get("overlapping_context_count")) is not int
        or any(
            not isinstance(receipt.get(k), str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", receipt[k])
            for k in ("context_labels_hash", "context_evidence_hash")
        )
    ):
        raise ValueError("complete context-disjoint critic provenance required")
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


def _validate_weighting_receipt(model: dict[str, Any], receipt: Any) -> None:
    if (
        not isinstance(receipt, dict)
        or receipt.get("loss_weighting_profile") != model["loss_weighting_profile"]
    ):
        raise ValueError("declared loss-weighting receipt required")
    if model["loss_weighting_profile"] == "uniform-frame":
        if any(
            k in receipt
            for k in (
                "sample_weighting",
                "event_label_hash",
                "event_evidence_hash",
                "event_partition_frame_counts",
            )
        ):
            raise ValueError("uniform objective cannot carry reweighting")
        return
    weighting = receipt.get("sample_weighting")
    event_profile = model["loss_weighting_profile"] == "equal-first-contact-lead-mass"
    counts = receipt.get("event_partition_frame_counts" if event_profile else "phase_frame_counts")
    if not event_profile and any(
        k in receipt
        for k in (
            "event_label_hash",
            "event_evidence_hash",
            "event_partition_frame_counts",
            "future_event_is_actor_observation",
        )
    ):
        raise ValueError("event credit cannot migrate to a phase-balanced receipt")
    if event_profile and (
        receipt.get("credit_lead_frames") != 8
        or receipt.get("credit_after_frames") != 20
        or receipt.get("future_event_is_actor_observation") is not False
        or receipt.get("future_event_used_only_as_offline_label") is not True
        or receipt.get("core_event_source_hash") != model["core_event_source_hash"]
        or any(
            not re.fullmatch(r"sha256:[0-9a-f]{64}", receipt.get(k, ""))
            for k in ("event_label_hash", "event_evidence_hash")
        )
    ):
        raise ValueError("complete offline event-credit provenance required")
    if (
        type(weighting) is not dict
        or weighting.get("schema") != "rosclaw.growth.positive_sample_weighting.v1"
        or weighting.get("row_count") != receipt.get("frame_sample_count")
        or type(weighting.get("row_count")) is not int
        or weighting.get("all_numeric_rows_retained") is not True
        or any(
            weighting.get(k) is not False
            for k in (
                "physical_batch_verified",
                "runtime_execution_authorized",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", weighting.get("sample_weight_hash", ""))
        or type(counts) is not list
        or len(counts) != (5 if event_profile else 3)
        or any(type(v) is not int or v < (0 if event_profile else 1) for v in counts)
        or sum(counts) != receipt.get("frame_sample_count")
        or not any(counts)
        or any(
            type(weighting.get(k)) is not float or not np.isfinite(weighting[k])
            for k in ("minimum", "maximum", "mean")
        )
        or not np.isclose(weighting["mean"], 1.0, atol=1e-12, rtol=0)
        or not 1 / 16 <= weighting["minimum"] <= weighting["maximum"] <= 16
        or weighting["minimum"] != sum(counts) / (sum(v > 0 for v in counts) * max(counts))
        or weighting["maximum"]
        != sum(counts) / (sum(v > 0 for v in counts) * min(v for v in counts if v > 0))
    ):
        raise ValueError("complete positive phase-balanced weighting receipt required")


def initial_model(
    actor: dict[str, Any],
    *,
    maximum_mean_kl: float,
    loss_weighting_profile: str = "uniform-frame",
    protected_domain_bank: dict[str, Any] | None = None,
    critic_profile: str = "whole-rollout",
) -> dict[str, Any]:
    validate_initial(actor)
    model = _initial_descriptor(
        copy.deepcopy(actor),
        maximum_mean_kl=maximum_mean_kl,
        loss_weighting_profile=loss_weighting_profile,
        protected_domain_bank=copy.deepcopy(protected_domain_bank),
        critic_profile=critic_profile,
    )
    # The public actor and trainable layers must not share mutable containers:
    # deepcopy of the whole descriptor would preserve that internal alias.
    model["residual_layers"] = copy.deepcopy(model["residual_layers"])
    model["model_hash"] = hash_json(model)
    validate_model(model)
    return model


def _initial_descriptor(
    actor: dict[str, Any],
    *,
    maximum_mean_kl: float,
    loss_weighting_profile: str,
    protected_domain_bank: dict[str, Any] | None = None,
    critic_profile: str = "whole-rollout",
) -> dict[str, Any]:
    """Internal commitment only; caller must validate actor before using it.

    No validation is cached or bypassed: validate_model already validates the
    complete initial actor before computing its expected parent commitment.
    Avoid recursively invoking public initial_model (which would validate the
    same actor twice again). The public builder still owns a full deep copy.
    """
    if critic_profile not in CRITIC_PROFILES:
        raise ValueError("explicit known critic profile required")
    extra: dict[str, Any] = {}
    if critic_profile == "whole-context":
        from rosclaw.growth import context_crossfit as context_module

        extra = dict(
            critic_profile=critic_profile,
            core_context_source_hash=hash_bytes(Path(context_module.__file__).read_bytes()),
        )
    elif critic_profile == "whole-context-neural":
        from rosclaw.growth import context_prediction_mlp, neural_context_advantages

        extra = dict(
            critic_profile=critic_profile,
            core_context_source_hash=hash_bytes(
                Path(neural_context_advantages.__file__).read_bytes()
            ),
            core_prediction_source_hash=hash_bytes(
                Path(context_prediction_mlp.__file__).read_bytes()
            ),
            neural_critic_fit_results=None,
        )
    return dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        generation=0,
        initial_actor=actor,
        parent_model_hash=actor["parent_model_hash"],
        residual_layers=actor["residual_layers"],
        raw_residual_cap=0.2,
        learning_rate=0.0004,
        maximum_mean_kl=maximum_mean_kl,
        proposal_execution_ceiling="PROPOSAL_ONLY_NO_RUNTIME",
        learning_receipt=None,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        core_regression_source_hash=hash_bytes(Path(regression_module.__file__).read_bytes()),
        core_index_source_hash=hash_bytes(Path(index_module.__file__).read_bytes()),
        core_weighting_source_hash=hash_bytes(Path(weighting_module.__file__).read_bytes()),
        core_domain_source_hash=hash_bytes(Path(domain_module.__file__).read_bytes()),
        core_event_source_hash=hash_bytes(Path(event_module.__file__).read_bytes()),
        protected_domain_bank=protected_domain_bank,
        loss_weighting_profile=loss_weighting_profile,
        memory_query_representation="EXACT_COORDINATE_INDEX_INTACT_LOGICAL_BANK",
        **dict.fromkeys(FLAGS, False),
        **extra,
    )


def make_preview(model: dict[str, Any]) -> dict[str, Any]:
    validate_model(model)
    policy = make_policy(np.zeros((3, 12)), 0.25, model["model_hash"])
    policy["execution_profile"] = "causal_per_frame_neural_residual"
    policy["step_motor_proof"] = dict(
        schema="soccer.rsi.proposal_memory_preview.v1",
        model=model,
        decision_start_frame=30,
        nominal_target_is_pre_motor=True,
        force_input="previous_completed_frame",
        execution_source_hash=model["source_hash"],
        qualification="UNQUALIFIED_SIM_PROPOSAL_REGRESSION",
        promotion_authorized=False,
    )
    policy["proposal_memory_motor_proof"] = dict(
        parent_model_hash=model["parent_model_hash"],
        memory_hash=model["initial_actor"]["baseline"]["memory"]["memory_hash"],
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledProposalMemoryMotor(CompiledCurrentMemoryMotor):
    def __init__(self, policy: dict[str, Any]) -> None:
        model = copy.deepcopy(policy["step_motor_proof"]["model"])
        if make_preview(model) != policy:
            raise ValueError("proposal-memory execution commitment changed")
        super().__init__(initial_preview(model["initial_actor"]))
        # Separate proposal family only: the original compiled actors and every
        # logical memory row/hash remain unchanged. Full base validation above
        # precedes construction of this opt-in exact-coordinate query index.
        self._parent._output_memory = IndexedAnchorOutputMemory.from_dict(
            self._parent._output_memory.to_dict()
        )
        self._layers = [
            (np.asarray(v["weight"], dtype=np.float64), np.asarray(v["bias"], dtype=np.float64))
            for v in model["residual_layers"]
        ]
        for w, b in self._layers:
            w.flags.writeable = b.flags.writeable = False
        self._zero = not np.any(self._layers[-1][0]) and not np.any(self._layers[-1][1])
        self._cap = 0.2
        if model["protected_domain_bank"] is not None:
            self._guard = DomainAnchorGuard(model["protected_domain_bank"]["bank"], bandwidth=1e-4)
        self._policy_hash = policy["policy_hash"]
