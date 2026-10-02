"""Cache the WHOLE smooth mean once; preserve every ordinary sampling hash.

Only persistence/construction is accelerated. The original smooth constructor
and decoder remain authoritative. No numerical field or random stream changes.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rosclaw.growth.frozen_payload_field import FrozenPayloadField
from rosclaw.growth.shared_proof_payload import MARKER, SCHEMA, canonical_hash, restore_payload

from rosclaw_soccer.rsi.kernel_guarded_step_network import FLAGS
from rosclaw_soccer.rsi.smooth_memory_motor import make_sampling_view
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once


def exact_shared_views(base: dict[str, Any], seeds: Iterable[int]) -> list[dict[str, Any]]:
    seeds = list(seeds)
    if (
        not 1 <= len(seeds) <= 52 * 16
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("distinct bounded declared exploration seeds required")
    original = hash_json(base)
    template = make_sampling_view(base, seed=seeds[0], std=0.1, rho=0.9)
    if template["mean_model"] != base:
        raise ValueError("authoritative constructor changed complete mean")
    cache = FrozenPayloadField(base)
    fields = {k: v for k, v in template.items() if k not in ("mean_model", "model_hash")}
    result = []
    for seed in seeds:
        unsigned = {**fields, "seed": seed}
        signed = {**unsigned, "model_hash": cache.document_hash(unsigned, "mean_model")}
        result.append(cache.envelope(signed, "mean_model"))
    # Full ordinary first/last constructors are independent witnesses, not
    # merely a self-consistency test of the new cached hasher.
    for index in sorted({0, len(seeds) - 1}):
        ordinary = (
            template
            if index == 0
            else make_sampling_view(base, seed=seeds[index], std=0.1, rho=0.9)
        )
        if restore_payload(result[index], base) != ordinary:
            raise ValueError("cached construction changed complete authoritative view")
    if hash_json(base) != original:
        raise ValueError("sampling construction changed immutable complete mean")
    return result


def publish_shared_views(root: Path, base: dict[str, Any], views: list[dict[str, Any]]) -> None:
    """Validate all complete logical seals before publishing a single payload."""
    models, store = root / "models", root / ".shared-models"
    if root.is_symlink() or models.is_symlink() or store.is_symlink() or not models.is_dir():
        raise ValueError("local new model directory required")
    if not 1 <= len(views) <= 52 * 16:
        raise ValueError("bounded sampling envelope batch required")
    cache = FrozenPayloadField(base)
    seeds = set()
    for view in views:
        if (
            type(view) is not dict
            or set(view)
            != {
                "schema",
                "location",
                "payload_hash",
                "logical_document_hash",
                "stripped_document",
                "envelope_hash",
            }
            or view["schema"] != SCHEMA
            or view["location"] != ["mean_model"]
            or view["payload_hash"] != cache.payload_hash
            or view["envelope_hash"]
            != canonical_hash({k: v for k, v in view.items() if k != "envelope_hash"})
        ):
            raise ValueError("exact sealed shared sampling envelope required")
        fields = dict(view["stripped_document"])
        if fields.pop("mean_model", None) != {MARKER: cache.payload_hash}:
            raise ValueError("complete mean marker required")
        seed = fields.get("seed")
        if type(seed) is not int or not 0 <= seed < 2**32 or seed in seeds:
            raise ValueError("distinct bounded exploration seed required")
        seeds.add(seed)
        if (
            fields.get("schema") != "soccer.rsi.smooth_memory_sampling.v1"
            or fields.get("std_raw") != 0.1
            or fields.get("rho") != 0.9
            or fields.get("activation_ceiling") != "SIM_ONLY"
            or fields.get("training_only") is not True
            or any(fields.get(k) is not False for k in FLAGS)
            or cache.document_hash(fields, "mean_model") != view["logical_document_hash"]
            or fields.get("model_hash")
            != cache.document_hash(
                {k: v for k, v in fields.items() if k != "model_hash"}, "mean_model"
            )
        ):
            raise ValueError("complete logical sampling seal changed")
    store.mkdir(exist_ok=True)
    payload_path = store / f"{cache.payload_hash[7:]}.json.gz"
    if payload_path.is_symlink():
        raise ValueError("complete mean cannot be a symlink")
    write_once(payload_path, base)
    for index, view in enumerate(views):
        path = models / f"sample-{index}.json.gz"
        if path.is_symlink():
            raise ValueError("sampling envelope cannot be a symlink")
        write_once(path, view)
