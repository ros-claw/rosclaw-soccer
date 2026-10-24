"""Explicit SIM-only admission for a frozen learned motor plus new neural head.

Existing dual-motor loaders retain their old return/type and budget contracts.
Only the explicit generic simulation dispatch admits this separately named kind.
"""

import json
import math
from pathlib import Path
from typing import Any

import torch

from rosclaw_soccer.rsi.consolidated_adapter_learning import (
    FEATURE_STATUS,
    KIND,
    STATUS,
    adapter_config,
    adapter_fit_plan,
    adapter_schedule,
    validate_head_checkpoint,
)
from rosclaw_soccer.rsi.consolidated_motor_adapter import ConsolidatedMotorAdapter
from rosclaw_soccer.rsi.dual_motor_checkpoint import digest
from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
from rosclaw_soccer.rsi.dual_recurrent_motor import DualRecurrentMotor
from rosclaw_soccer.rsi.paid_full_entry_student import FrozenFullEntryActor
from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter
from rosclaw_soccer.rsi.refinement_motor_learning import KIND as REFINEMENT_KIND
from rosclaw_soccer.rsi.supported_refinement_motor import SupportedRefinementMotor


def load_consolidated_adapter(
    parent: FrozenFullEntryActor,
    *,
    parent_lineage_hash: str,
    root: Path,
    expected_receipt_hash: str,
) -> tuple[ConsolidatedMotorAdapter, dict[str, Any]]:
    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        root.is_symlink()
        or any(p.is_symlink() for p in (rp, dp))
        or (root / "failed.json").exists()
        or digest(rp) != expected_receipt_hash
    ):
        raise ValueError("exact completed new-head training required")
    r, d = [json.loads(p.read_text()) for p in (rp, dp)]
    cfg = adapter_config(r["config"]["seed"], r["config"]["device"])
    identities = (
        "source_head",
        "data_hash",
        "feature_hash",
        "fit_plan_hash",
        "memory_training_hash",
        "memory_checkpoint_hash",
        "parent_lineage_hash",
        "fit_plan",
        "input_file_hashes",
    )
    if (
        r["status"] != STATUS
        or r["checkpoint_kind"] != KIND
        or r["declaration_hash"] != digest(dp)
        or r["config"] != cfg
        or d["config"] != cfg
        or any(r[k] != d[k] for k in identities)
        or r["parent_lineage_hash"] != parent_lineage_hash
        or r["head_optimizer_steps"] != cfg["head_steps"]
        or r["memory_optimizer_steps_unchanged"] != dict(actor=112, critic=48)
        or r["physical_steps_executed"] != 0
        or r["old_retention_proven"] is not False
        or r["online_RL_claim"] is not False
        or any(
            x[k] is not False
            for x in (r, d)
            for k in ("goal_complete", "hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("fixed actual new-head2048/unchanged memory112/48 contract required")

    def receipt(h: str) -> tuple[Path, dict[str, Any]]:
        paths = [
            Path(p)
            for p, v in r["input_file_hashes"].items()
            if v == h and p.endswith("/complete.json")
        ]
        if len(paths) != 1 or paths[0].is_symlink() or digest(paths[0]) != h:
            raise ValueError("unique exact learned-memory and own-feature ancestry required")
        return paths[0], json.loads(paths[0].read_text())

    _, f = receipt(r["feature_hash"])
    _, bank = receipt(r["data_hash"])
    mp, memory_receipt = receipt(r["memory_training_hash"])
    if (
        f["status"] != FEATURE_STATUS
        or f["data_hash"] != r["data_hash"]
        or f["memory_training_hash"] != r["memory_training_hash"]
        or f["memory_checkpoint_hash"] != r["memory_checkpoint_hash"]
        or f["zero_head_bit_identity"] is not True
        or f["teacher_hidden_state_supplied"] is not False
        or len(f["records"]) != 274
        or f["own_feature_decisions"] != sum(x["rows"] for x in bank["records"])
        or bank["parent_lineage_hash"] != parent_lineage_hash
        or memory_receipt["checkpoint_hash"] != r["memory_checkpoint_hash"]
        or memory_receipt["cumulative_optimizer_steps"] != 112
        or memory_receipt["cumulative_critic_optimizer_steps"] != 48
    ):
        raise ValueError("complete actual own-feature bank bound to entire learned memory required")
    plan = adapter_fit_plan(bank["records"])
    if r["fit_plan"] != plan or r["fit_plan_hash"] != plan["hash"]:
        raise ValueError("original17 slow-memory anchors and coherent raw-label plan required")
    schedule = adapter_schedule(plan, cfg["seed"])
    if len(r["history"]) != len(schedule):
        raise ValueError("all2048 actual optimizer history entries required")
    for actual, expected in zip(r["history"], schedule, strict=True):
        if any(actual[k] != v for k, v in expected.items()) or any(
            type(actual[k]) not in (float, int) or not math.isfinite(actual[k])
            for k in ("loss", "imitation", "retention", "baseline", "gradient_norm")
        ):
            raise ValueError("complete seeded real-history schedule and finite losses required")
    memory, _ = load_dual_candidate_for_sim_audit(
        parent,
        parent_lineage_hash=parent_lineage_hash,
        root=mp.parent,
        expected_receipt_hash=r["memory_training_hash"],
        checkpoint_kind="contact-consolidated-v2",
    )
    model = ConsolidatedMotorAdapter(memory, head_bound_rad=cfg["head_bound_rad"])
    cp = Path(r["checkpoint_path"])
    if (
        cp.is_symlink()
        or cp.parent.resolve() != root.resolve()
        or cp.stat().st_size > 16000000
        or digest(cp) != r["checkpoint_hash"]
    ):
        raise ValueError("confined bounded exact tensor-only head checkpoint required")
    state = torch.load(cp, map_location="cpu", weights_only=True)
    ids = {
        k: r[k]
        for k in (
            "feature_hash",
            "data_hash",
            "fit_plan_hash",
            "memory_training_hash",
            "memory_checkpoint_hash",
            "parent_lineage_hash",
        )
    }
    validate_head_checkpoint(state, dict(model.head.state_dict()), config=cfg, commitments=ids)
    model.head.load_state_dict(state["head"], strict=True)
    return model.eval().requires_grad_(False), r


def load_motor_candidate_for_sim_audit(
    parent: FrozenFullEntryActor,
    *,
    parent_lineage_hash: str,
    root: Path,
    expected_receipt_hash: str,
    checkpoint_kind: str = "supervised",
) -> tuple[
    DualRecurrentMotor
    | ConsolidatedMotorAdapter
    | RefinementMotorAdapter
    | SupportedRefinementMotor,
    dict[str, Any],
]:
    if checkpoint_kind == "support-masked-refinement-v1":
        from rosclaw_soccer.rsi.supported_motor_checkpoint import load_supported_motor

        return load_supported_motor(
            parent,
            parent_lineage_hash=parent_lineage_hash,
            root=root,
            expected_receipt_hash=expected_receipt_hash,
        )
    if checkpoint_kind == REFINEMENT_KIND:
        from rosclaw_soccer.rsi.refinement_motor_checkpoint import load_refinement_motor

        return load_refinement_motor(
            parent,
            parent_lineage_hash=parent_lineage_hash,
            root=root,
            expected_receipt_hash=expected_receipt_hash,
        )
    if checkpoint_kind == KIND:
        return load_consolidated_adapter(
            parent,
            parent_lineage_hash=parent_lineage_hash,
            root=root,
            expected_receipt_hash=expected_receipt_hash,
        )
    return load_dual_candidate_for_sim_audit(
        parent,
        parent_lineage_hash=parent_lineage_hash,
        root=root,
        expected_receipt_hash=expected_receipt_hash,
        checkpoint_kind=checkpoint_kind,
    )


def expected_cpu_audit_decisions(training: dict[str, Any], checkpoint_kind: str) -> int:
    """Separate actual train-time replay counts from predeclared feature counts.

    New-head training does NOT claim recurrent CPU replay. Its separate audit
    must cover the full independently qualified input bank instead.
    """
    if checkpoint_kind == "support-masked-refinement-v1":
        from rosclaw_soccer.rsi.supported_motor_checkpoint import support_cpu_scope

        return support_cpu_scope(training)
    if checkpoint_kind == REFINEMENT_KIND:
        from rosclaw_soccer.rsi.refinement_motor_learning import FEATURE_STATUS as REF_FEATURE

        ps = [
            Path(p)
            for p, h in training["input_file_hashes"].items()
            if h == training["feature_hash"] and p.endswith("/complete.json")
        ]
        if len(ps) != 1 or ps[0].is_symlink() or digest(ps[0]) != training["feature_hash"]:
            raise ValueError("exact whole refinement feature proof required")
        f = json.loads(ps[0].read_text())
        count = f["own_feature_decisions"]
        if (
            f["status"] != REF_FEATURE
            or f["bank_hash"] != training["bank_hash"]
            or type(count) is not int
            or count != sum(r["rows"] for r in f["records"])
        ):
            raise ValueError("whole refinement private-history decision scope required")
        return count
    if checkpoint_kind != KIND:
        count = training["cpu_offline_decisions"]
        if type(count) is not int:
            raise ValueError("explicit actual legacy CPU decision count required")
        return count
    paths = [
        Path(p)
        for p, h in training["input_file_hashes"].items()
        if h == training["feature_hash"] and p.endswith("/complete.json")
    ]
    if len(paths) != 1 or paths[0].is_symlink() or digest(paths[0]) != training["feature_hash"]:
        raise ValueError("exact complete own-feature witness required for CPU audit scope")
    f = json.loads(paths[0].read_text())
    count = f["own_feature_decisions"]
    if (
        f["status"] != FEATURE_STATUS
        or len(f["records"]) != 274
        or type(count) is not int
        or count != sum(r["rows"] for r in f["records"])
        or f["data_hash"] != training["data_hash"]
    ):
        raise ValueError("all274 actual history decisions required")
    return count


def training_data_hash(training: dict[str, Any], checkpoint_kind: str) -> str:
    """New training data stays bank62, never mislabeled as the old task witness."""
    new = checkpoint_kind in (REFINEMENT_KIND, "support-masked-refinement-v1")
    value = training["bank_hash"] if new else training["data_hash"]
    if type(value) is not str or (new and (not value.startswith("sha256:") or len(value) != 71)):
        raise ValueError("explicit exact training-data commitment required")
    if new and training.get("data_hash", value) != value:
        raise ValueError("contradictory new bank identity")
    return value
