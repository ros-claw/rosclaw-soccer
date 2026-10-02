"""Explicit current-parent output consolidation around an existing learned actor.

The old actor, optimizer and frozen NN are unchanged. This is a separately
sealed post-learning memory ablation, not a new gradient update or promotion.
"""

import copy
from pathlib import Path
from typing import Any

import numpy as np
import rosclaw.growth.anchor_output_memory as memory_module
from rosclaw.growth.anchor_output_memory import AnchorOutputMemory

from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.kernel_guarded_step_network import FLAGS
from rosclaw_soccer.rsi.output_memory_step_motor import encoder_identity
from rosclaw_soccer.rsi.smooth_memory_motor import CompiledSmoothMemoryMotor
from rosclaw_soccer.rsi.smooth_memory_motor import make_preview as base_preview
from rosclaw_soccer.rsi.smooth_memory_motor import validate_model as validate_base
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.consolidated_smooth_motor.v1"
EVIDENCE_KEYS = (
    "schema",
    "parent_model_hash",
    "encoder_hash",
    "inherited_memory_hash",
    "bank_summary_hash",
    "bank_review_hash",
    "selection",
    "records",
    "source_hash",
    "consolidation_source_hash",
    "promotion_authorized",
    "hardware_authorized",
)


def validate_model(model: dict[str, Any]) -> AnchorOutputMemory:
    if (
        model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("adaptation_kind") != "CURRENT_PARENT_OUTPUT_CONSOLIDATION_NOT_NEW_OPTIMIZER"
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or model.get("core_memory_source_hash")
        != hash_bytes(Path(memory_module.__file__).read_bytes())
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
        or any(model.get(k) is not False for k in FLAGS)
    ):
        raise ValueError("sealed SIM-only current-parent consolidation ablation required")
    base, manifest = model["base_model"], model["consolidation_manifest"]
    validate_base(base)
    memory = AnchorOutputMemory.from_dict(model["memory"])
    parent = base["frozen_parent"]
    mapping = manifest["consolidation_mapping"]
    if (
        manifest["report_hash"]
        != hash_json({k: v for k, v in manifest.items() if k != "report_hash"})
        or manifest["schema"] != "soccer.rsi.current_parent_success_consolidation_evidence.v1"
        or manifest["selection"] != "ALL_QUALIFIED_CURRENT_PARENT_SUCCESSES_IN_SEALED_ORDER"
        or memory.parent_policy_hash != parent["model_hash"]
        or manifest["parent_model_hash"] != parent["model_hash"]
        or memory.encoder_hash != encoder_identity(parent["frozen_parent"])
        or memory.evidence_hash != hash_json({k: manifest[k] for k in EVIDENCE_KEYS})
        or manifest["memory_hash"] != model["memory"]["memory_hash"]
        or manifest["inherited_memory_hash"] != parent["output_memory"]["memory_hash"]
        or model["memory"]["predecessor_memory_hash"] != manifest["inherited_memory_hash"]
        or manifest["exact_current_parent_output_reload"] is not True
        or manifest["physical_policy_execution_qualified"] is not False
        or manifest["actual_physical_executions_added"] != 0
        or not 1 <= len(manifest["records"]) <= 52
        or manifest["independent_contexts"] != len(manifest["records"])
        or manifest["existing_success_reports_reconstructed"] != len(manifest["records"])
        or len({(r["seed"], r["lane"]) for r in manifest["records"]}) != len(manifest["records"])
        or any(r["frames"] != 270 for r in manifest["records"])
        or manifest["added_frames"] != sum(r["frames"] for r in manifest["records"])
        or manifest["added_frames"] != len(mapping["sample_to_memory_row"])
        or mapping["input_sample_count"] != manifest["added_frames"]
        or mapping["inherited_rows"] != manifest["inherited_frames"]
        or manifest["recorded_frames"] != len(model["memory"]["observations"])
        or mapping["inherited_memory_hash"] != manifest["inherited_memory_hash"]
        or mapping["consolidated_memory_hash"] != manifest["memory_hash"]
        or manifest["recorded_frames"]
        != manifest["inherited_frames"] + mapping["appended_unique_rows"]
        or mapping["distinct_inherited_rows_evicted"] != 0
        or mapping["rounding_used"] is not False
        or mapping["exact_duplicate_only"] is not True
        or any(
            type(i) is not int or not 0 <= i < manifest["recorded_frames"]
            for i in mapping["sample_to_memory_row"]
        )
        or any(
            obj.get(k) is not False
            for obj in (manifest, mapping)
            for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError(
            "complete all-success memory bound to the actual current NN parent required"
        )
    return memory


def make_model(
    base: dict[str, Any], memory: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    model = dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        adaptation_kind="CURRENT_PARENT_OUTPUT_CONSOLIDATION_NOT_NEW_OPTIMIZER",
        base_model=copy.deepcopy(base),
        memory=copy.deepcopy(memory),
        consolidation_manifest=copy.deepcopy(manifest),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        core_memory_source_hash=hash_bytes(Path(memory_module.__file__).read_bytes()),
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
        schema="soccer.rsi.consolidated_smooth_preview.v1",
        model=model,
        decision_start_frame=30,
        nominal_target_is_pre_motor=True,
        force_input="previous_completed_frame",
        execution_source_hash=model["source_hash"],
        qualification="UNQUALIFIED_SIM_CONSOLIDATION_ABLATION",
        promotion_authorized=False,
    )
    policy["consolidated_smooth_motor_proof"] = dict(
        memory_hash=model["memory"]["memory_hash"],
        parent_model_hash=model["base_model"]["parent_model_hash"],
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledConsolidatedSmoothMotor(CompiledSmoothMemoryMotor):
    def __init__(self, policy: dict[str, Any]) -> None:
        model = policy["step_motor_proof"]["model"]
        memory = validate_model(model)
        if make_preview(model) != policy:
            raise ValueError("consolidated execution commitment changed")
        super().__init__(base_preview(model["base_model"]))
        self._consolidated = memory
        self._policy_hash = policy["policy_hash"]

    def raw_mean(self, observation: Any, phase: int) -> Any:
        proposal = super().raw_mean(observation, phase)
        if self._zero:
            # Zero-addition inheritance remains globally exact, not just at
            # stored states. No new sampled policy is claimed by this wrapper.
            return proposal
        context = np.concatenate((self.features(observation)[:134], [phase]))
        return self._consolidated.blend(
            context, proposal, encoder_hash=self._consolidated.encoder_hash
        )
