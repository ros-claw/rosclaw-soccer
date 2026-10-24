"""Explicit tensor-only ancestry and bounded JSON neural-option bundle loader.

Admission here authorizes only the existing simulation exams, never promotion,
hardware, trained Option-Critic or unseen-course performance claims.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.committed_neural_motor import CommittedNeuralMotor
from rosclaw_soccer.rsi.dual_motor_checkpoint import digest
from rosclaw_soccer.rsi.neural_option_contract import (
    IDENTITIES,
    KIND,
    STATUS,
    option_config,
    option_cpu_scope,
)
from rosclaw_soccer.rsi.neural_option_initiation import NeuralOptionInitiation
from rosclaw_soccer.rsi.paid_full_entry_student import FrozenFullEntryActor
from rosclaw_soccer.rsi.supported_motor_checkpoint import support_receipt
from rosclaw_soccer.sim.contracts import hash_json


def option_ancestry(r: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], Path, list[Path]]:
    _, own = support_receipt(r, r["own97_hash"])
    fp, feasibility = support_receipt(r, r["feasibility_hash"])
    pp, parent = support_receipt(r, r["parent_training_hash"])
    if (
        option_cpu_scope(own) != 36276
        or own["feasibility_hash"] != r["feasibility_hash"]
        or feasibility["status"]
        != "NEURAL_OPTION_COVERAGE_AND_OWN_INITIAL_FEATURES_AUDITED_NOT_COMPOSED"
        or feasibility["contradictory_initial_feature_labels"] != 0
        or feasibility["initial_feature_rows"] != 97
        or feasibility["only_own_initial_features_used"] is not True
        or feasibility["state_id_task_id_are_provenance_not_selector_inputs"] is not True
        or feasibility["parent_training_hash"] != r["parent_training_hash"]
        or feasibility["initiation_hash"] != r["initiation_hash"]
        or parent["checkpoint_kind"] != "consolidated-adapter-v1"
        or parent["checkpoint_hash"] != r["parent_checkpoint_hash"]
        or parent["parent_lineage_hash"] != r["parent_lineage_hash"]
        or parent["data_hash"] != r["data_hash"]
        or len(r["expert_training_hashes"]) != 3
        or len(r["expert_checkpoint_hashes"]) != 3
        or len(set(r["expert_training_hashes"])) != 3
        or len(feasibility["empirical_potential"]["plan"]) != 80
        or [x["option"] for x in own["records"][:80]]
        != [x["option_label"] for x in feasibility["empirical_potential"]["plan"]]
    ):
        raise ValueError("exact same-parent own97/initiation97/three-expert lineage required")
    expert_paths = []
    for index, (h, ch) in enumerate(
        zip(r["expert_training_hashes"], r["expert_checkpoint_hashes"], strict=True)
    ):
        tp, trained = support_receipt(r, h)
        if (
            trained["checkpoint_kind"] != "consolidated-refinement-v1"
            or trained["config"]["seed"] != 115023 + index
            or trained["new_head_steps"] != 2048
            or trained["parent_training_hash"] != r["parent_training_hash"]
            or trained["parent_lineage_hash"] != r["parent_lineage_hash"]
            or trained["checkpoint_hash"] != ch
            or own["input_file_hashes"].get(str(tp)) != h
        ):
            raise ValueError("three exact independently qualified learned neural experts required")
        expert_paths.append(tp.parent)
    generation = str(
        hash_json(
            dict(
                schema="prototype.committed_neural_motor.v1",
                source_head=own["source_head"],
                feasibility_hash=r["feasibility_hash"],
                parent_training_hash=r["parent_training_hash"],
                experts={i + 1: h for i, h in enumerate(r["expert_training_hashes"])},
            )
        )
    )
    if generation != r["generation_hash"] or generation != own["generation_hash"]:
        raise ValueError("exact independently replayed model-generation commitment required")
    return own, feasibility, pp.parent, expert_paths


def load_committed_motor(
    parent: FrozenFullEntryActor,
    *,
    parent_lineage_hash: str,
    root: Path,
    expected_receipt_hash: str,
) -> tuple[CommittedNeuralMotor, dict[str, Any]]:
    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        root.is_symlink()
        or any(p.is_symlink() for p in (rp, dp))
        or (root / "failed.json").exists()
        or digest(rp) != expected_receipt_hash
    ):
        raise ValueError("exact completed ordinary neural-option reference bundle required")
    r, d = [json.loads(p.read_text()) for p in (rp, dp)]
    if (
        r["status"] != STATUS
        or r["checkpoint_kind"] != KIND
        or r["declaration_hash"] != digest(dp)
        or r["parent_lineage_hash"] != parent_lineage_hash
        or r["config"] != option_config()
        or r["checkpoint_format"] != "json.reference_bundle.v1"
        or any(
            r[k] != d[k]
            for k in (
                *IDENTITIES,
                "source_head",
                "input_file_hashes",
                "config",
                "checkpoint_hash",
                "checkpoint_path",
            )
        )
        or r["physical_steps_executed"] != 0
        or r["new_optimizer_updates"] != 0
        or r["old_retention_proven"] is not False
        or r["online_RL_claim"] is not False
        or any(
            v[k] is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError(
            "read-only SIM composition cannot fabricate training or physical permission"
        )
    cp = Path(r["checkpoint_path"])
    if (
        cp.is_symlink()
        or cp.resolve().parent != root.resolve()
        or cp.stat().st_size > 16384
        or digest(cp) != r["checkpoint_hash"]
    ):
        raise ValueError("confined bounded exact neural-option JSON descriptor required")
    if json.loads(cp.read_text()) != dict(schema=KIND, **{k: r[k] for k in IDENTITIES}):
        raise ValueError("all neural expert, own-state geometry and generation identities required")
    _, feasibility, parent_root, expert_roots = option_ancestry(r)
    fp, _ = support_receipt(r, r["feasibility_hash"])
    ip = fp.parent / feasibility["initiation_path"]
    if (
        ip.is_symlink()
        or ip.resolve().parent != fp.parent.resolve()
        or ip.stat().st_size > 8 * 1024**2
        or digest(ip) != r["initiation_hash"]
    ):
        raise ValueError("confined bounded exact own-feature initiation archive required")
    with np.load(ip, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    if arrays["features"].shape != (97, 288) or not np.array_equal(
        arrays["labels"],
        np.array(
            [x["option_label"] for x in feasibility["empirical_potential"]["plan"]] + [0] * 17,
            dtype=np.int64,
        ),
    ):
        raise ValueError("all own97 feature labels must match the qualified evidence plan")
    initiation = NeuralOptionInitiation(
        torch.from_numpy(arrays["features"]), torch.from_numpy(arrays["labels"])
    )
    if not torch.equal(initiation.scale, torch.from_numpy(arrays["scale"])) or not torch.equal(
        initiation.radii, torch.from_numpy(arrays["prospective_radius"])
    ):
        raise ValueError("independently recomputed initiation geometry differs")
    from rosclaw_soccer.rsi.consolidated_adapter_checkpoint import load_consolidated_adapter
    from rosclaw_soccer.rsi.refinement_motor_checkpoint import load_refinement_motor

    acquired, _ = load_consolidated_adapter(
        parent,
        parent_lineage_hash=parent_lineage_hash,
        root=parent_root,
        expected_receipt_hash=r["parent_training_hash"],
    )
    experts = {}
    for index, (path, h) in enumerate(
        zip(expert_roots, r["expert_training_hashes"], strict=True), start=1
    ):
        experts[index], _ = load_refinement_motor(
            parent, parent_lineage_hash=parent_lineage_hash, root=path, expected_receipt_hash=h
        )
    return CommittedNeuralMotor(
        acquired, experts, initiation, generation_hash=r["generation_hash"]
    ), r
