"""Lossless new-seed views; metadata registry, not physically collected data."""

from typing import Any

from rosclaw.growth.frozen_payload_field import FrozenPayloadField
from rosclaw.growth.shared_proof_payload import restore_payload

from rosclaw_soccer.rsi.positive_lateral_receding_scope import build_scope
from rosclaw_soccer.sim.contracts import hash_json


def registry_views(
    scope: dict[str, Any], source_envelope: dict[str, Any], mean: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    """Whole ordinary source restoration, unchanged mean and sampling law.

    Caller authenticates source receipt/file bytes and stores the exact payload.
    No model constructor, NN, physical body, result filtering or authority.
    """
    if mean.get("model_hash") != hash_json(
        {k: v for k, v in mean.items() if k != "model_hash"}
    ) or scope != build_scope(mean["model_hash"]):
        raise ValueError("exact source mean and complete prospective scope required")
    original = restore_payload(source_envelope, mean)
    if (
        original.get("schema") != "soccer.rsi.recurrent_motor_sampling.v1"
        or original.get("std_raw") != 0.1
        or original.get("rho") != 0.9
        or original.get("model_hash")
        != hash_json({k: v for k, v in original.items() if k != "model_hash"})
        or original.get("activation_ceiling") != "SIM_ONLY"
        or original.get("training_only") is not True
        or any(
            original.get(k) is not False
            for k in (
                "promotion_authorized",
                "runtime_execution_authorized",
                "hardware_authorized",
                "fresh_holdout_open_authorized",
            )
        )
    ):
        raise ValueError("whole original frozen recurrent sampling law required")
    cache = FrozenPayloadField(mean)
    fields = {k: v for k, v in original.items() if k not in ("mean_model", "seed", "model_hash")}
    result = {}
    for context in scope["contexts"]:
        unsigned = dict(fields, seed=context["sampling_seed"])
        signed = dict(unsigned, model_hash=cache.document_hash(unsigned, "mean_model"))
        envelope = cache.envelope(signed, "mean_model")
        # An ordinary canonical reconstruction, not just agreement between two
        # cached hashes. This does not instantiate or execute the policy.
        if context["context_index"] in (0, 7):
            ordinary = dict(unsigned, mean_model=mean)
            ordinary["model_hash"] = hash_json(ordinary)
            if restore_payload(envelope, mean) != ordinary:
                raise ValueError("new seed view must equal complete ordinary construction")
        result[context["source_group"]] = envelope
    if len(result) != 8:
        raise ValueError("all eight prospective source views required")
    return result
