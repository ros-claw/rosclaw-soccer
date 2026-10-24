"""Explicit SIM-only reference-bundle admission; never unpickle a new policy."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rosclaw_soccer.rsi.dual_motor_checkpoint import digest
from rosclaw_soccer.rsi.feature_support_gate import FeatureSupportGate
from rosclaw_soccer.rsi.paid_full_entry_student import FrozenFullEntryActor
from rosclaw_soccer.rsi.supported_refinement_motor import SupportedRefinementMotor

KIND = "support-masked-refinement-v1"
STATUS = "SUPPORTED_NEURAL_REFERENCE_BUNDLE_ASSEMBLED_PENDING_CPU_AND_PHYSICS"
IDENTITIES = (
    "source_refinement_training_hash",
    "source_refinement_checkpoint_hash",
    "parent_training_hash",
    "parent_lineage_hash",
    "bank_hash",
    "feature_hash",
    "retention_feature_hash",
    "geometry_hash",
    "gate_audit_hash",
)


def support_receipt(r: dict[str, Any], h: str) -> tuple[Path, dict[str, Any]]:
    paths = [
        Path(p)
        for p, v in r["input_file_hashes"].items()
        if v == h and p.endswith("/complete.json")
    ]
    if (
        len(paths) != 1
        or paths[0].is_symlink()
        or digest(paths[0]) != h
        or (paths[0].parent / "failed.json").exists()
    ):
        raise ValueError("unique exact closed support-model ancestry required")
    return paths[0], json.loads(paths[0].read_text())


def support_cpu_scope(r: dict[str, Any]) -> int:
    _, f = support_receipt(r, r["feature_hash"])
    _, old = support_receipt(r, r["retention_feature_hash"])
    if (
        f["status"] != "PASS_REFINEMENT_OWN_FEATURES_ZERO_IDENTITY_NOT_PHYSICS"
        or old["status"]
        != "CURRENT_PARENT_RETENTION_OWN_FEATURES_MATCH_ACTUAL_PHYSICS_NOT_NEW_LEARNING"
        or f["bank_hash"] != r["bank_hash"]
        or old["feature_hash"] != r["feature_hash"]
        or f["parent_training_hash"] != r["parent_training_hash"]
        or old["parent_training_hash"] != r["parent_training_hash"]
        or len(f["records"]) != 62
        or len(old["records"]) != 17
        or f["own_feature_decisions"] != sum(x["rows"] for x in f["records"])
        or old["own_actor_decisions"] != sum(x["rows"] for x in old["records"])
    ):
        raise ValueError("complete same-parent62+17 own-feature scope required")
    return int(f["own_feature_decisions"] + old["own_actor_decisions"])


def load_supported_motor(
    parent: FrozenFullEntryActor,
    *,
    parent_lineage_hash: str,
    root: Path,
    expected_receipt_hash: str,
) -> tuple[SupportedRefinementMotor, dict[str, Any]]:
    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        root.is_symlink()
        or any(p.is_symlink() for p in (rp, dp))
        or (root / "failed.json").exists()
        or digest(rp) != expected_receipt_hash
    ):
        raise ValueError("exact completed support reference bundle required")
    r, d = [json.loads(p.read_text()) for p in (rp, dp)]
    if (
        r["status"] != STATUS
        or r["checkpoint_kind"] != KIND
        or r["declaration_hash"] != digest(dp)
        or r["parent_lineage_hash"] != parent_lineage_hash
        or r["config"]
        != dict(
            seed=r["config"]["seed"],
            new_optimizer_updates=0,
            reused_head_updates=2048,
            support_schema=1,
        )
        or r["config"]["seed"] not in (115023, 115024, 115025)
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
        or r["checkpoint_format"] != "json.reference_bundle.v1"
        or r["new_optimizer_updates"] != 0
        or r["physical_steps_executed"] != 0
        or r["old_retention_proven"] is not False
        or any(
            v[k] is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("read-only SIM support assembly cannot fabricate training or permission")
    cp = Path(r["checkpoint_path"])
    if (
        cp.is_symlink()
        or cp.resolve().parent != root.resolve()
        or cp.stat().st_size > 16384
        or digest(cp) != r["checkpoint_hash"]
    ):
        raise ValueError("confined bounded exact JSON reference descriptor required")
    descriptor = json.loads(cp.read_text())
    if descriptor != dict(schema=KIND, **{k: r[k] for k in IDENTITIES}):
        raise ValueError("full neural-weight and support commitments required")
    tp, trained = support_receipt(r, r["source_refinement_training_hash"])
    gp, geometry = support_receipt(r, r["geometry_hash"])
    _, gate = support_receipt(r, r["gate_audit_hash"])
    if (
        trained["checkpoint_kind"] != "consolidated-refinement-v1"
        or trained["checkpoint_hash"] != r["source_refinement_checkpoint_hash"]
        or trained["config"]["seed"] != r["config"]["seed"]
        or any(
            trained[k] != r[k]
            for k in ("parent_training_hash", "parent_lineage_hash", "bank_hash", "feature_hash")
        )
        or geometry["retention_feature_hash"] != r["retention_feature_hash"]
        or geometry["feature_hash"] != r["feature_hash"]
        or gate["status"] != "ALL_PROTECTED_FEATURES_ZERO_RESCUE_ANCHORS_ONE_GATE_AUDITED_NOT_MOTOR"
        or gate["geometry_hash"] != r["geometry_hash"]
        or gate["protected_bitwise_zero"] is not True
        or gate["corrective_bitwise_one"] is not True
        or gate["own_geometry_independently_recomputed"] is not True
        or support_cpu_scope(r) != 28977
    ):
        raise ValueError("same trained head and complete qualified zero/one support required")
    ap = gp.parent / geometry["geometry_path"]
    if (
        ap.is_symlink()
        or ap.resolve().parent != gp.parent.resolve()
        or ap.stat().st_size > 128 * 1024**2
        or digest(ap) != geometry["geometry_hash"]
    ):
        raise ValueError("confined qualified geometry archive required")
    with np.load(ap, allow_pickle=False) as z:
        support = FeatureSupportGate(
            torch.from_numpy(z["corrective_features"].copy()),
            torch.from_numpy(z["scale"].copy()),
            torch.from_numpy(z["prospective_radius"].copy()),
        )
    from rosclaw_soccer.rsi.refinement_motor_checkpoint import load_refinement_motor

    learned, _ = load_refinement_motor(
        parent,
        parent_lineage_hash=parent_lineage_hash,
        root=tp.parent,
        expected_receipt_hash=r["source_refinement_training_hash"],
    )
    return SupportedRefinementMotor(learned, support), r
