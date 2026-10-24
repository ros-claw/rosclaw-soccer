"""Strict SIM-only new-type admission; old acquired-head validators stay intact."""

import json
import math
from pathlib import Path
from typing import Any

import torch

from rosclaw_soccer.rsi.consolidated_adapter_checkpoint import load_consolidated_adapter
from rosclaw_soccer.rsi.dual_motor_checkpoint import digest
from rosclaw_soccer.rsi.paid_full_entry_student import FrozenFullEntryActor
from rosclaw_soccer.rsi.pass_refinement_curriculum import refinement_curriculum
from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter
from rosclaw_soccer.rsi.refinement_motor_learning import (
    COMMITMENTS,
    FEATURE_STATUS,
    KIND,
    STATUS,
    refinement_config,
    refinement_schedule,
    validate_refinement_checkpoint,
)
from rosclaw_soccer.sim.contracts import hash_json


def load_refinement_motor(
    parent: FrozenFullEntryActor,
    *,
    parent_lineage_hash: str,
    root: Path,
    expected_receipt_hash: str,
) -> tuple[RefinementMotorAdapter, dict[str, Any]]:
    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        root.is_symlink()
        or any(p.is_symlink() for p in (rp, dp))
        or (root / "failed.json").exists()
        or digest(rp) != expected_receipt_hash
    ):
        raise ValueError("exact completed new-type neural refinement required")
    r, d = [json.loads(p.read_text()) for p in (rp, dp)]
    cfg = refinement_config(r["config"]["seed"], r["config"]["device"])
    if (
        r["status"] != STATUS
        or r["checkpoint_kind"] != KIND
        or r["declaration_hash"] != digest(dp)
        or r["parent_lineage_hash"] != parent_lineage_hash
        or r["config"] != cfg
        or d["config"] != cfg
        or d["preflight_only"] is not False
        or any(r[k] != d[k] for k in COMMITMENTS | {"source_head", "input_file_hashes", "fit_plan"})
        or r["new_head_steps"] != 2048
        or r["acquired_head_steps_unchanged"] != 2048
        or r["memory_steps_unchanged"] != dict(actor=112, critic=48)
        or r["parent_weights_unchanged"] is not True
        or r["physical_steps_executed"] != 0
        or r["physical_retention_proven"] is not False
        or r["online_RL_claim"] is not False
        or any(
            v[k] is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("exact whole-parent-frozen2048 new-head contract required")

    def receipt(h: str) -> tuple[Path, dict[str, Any]]:
        paths = [
            Path(p)
            for p, v in r["input_file_hashes"].items()
            if v == h and p.endswith("/complete.json")
        ]
        if len(paths) != 1 or paths[0].is_symlink() or digest(paths[0]) != h:
            raise ValueError("unique exact refinement ancestry required")
        return paths[0], json.loads(paths[0].read_text())

    _, f = receipt(r["feature_hash"])
    _, bank = receipt(r["bank_hash"])
    pp, acquired = receipt(r["parent_training_hash"])
    _, coverage = receipt(bank["coverage_hash"])
    _, native = receipt(bank["native_hash"])
    _, old = receipt(bank["retained_bank_hash"])
    plan = refinement_curriculum(coverage, native["records"], old["records"])
    if (
        f["status"] != FEATURE_STATUS
        or f["bank_hash"] != r["bank_hash"]
        or f["parent_training_hash"] != r["parent_training_hash"]
        or f["parent_checkpoint_hash"] != r["parent_checkpoint_hash"]
        or f["zero_refinement_bit_identity"] is not True
        or f["all_shared_parent_traces_match"] is not True
        or f["teacher_hidden_state_executed"] is not False
        or f["parent_weights_unchanged"] is not True
        or f["own_feature_decisions"] != sum(x["rows"] for x in bank["records"])
        or len(f["records"]) != len(bank["records"])
        or any(v["fit_plan"] != plan for v in (r, f, bank))
        or bank["current80_outcomes"] != native["records"]
        or bank["original274_metadata"] != old["records"]
        or acquired["checkpoint_hash"] != r["parent_checkpoint_hash"]
        or acquired["config"]["seed"] != 105025
        or acquired["head_optimizer_steps"] != 2048
    ):
        raise ValueError("whole private feature and real-rescue/protected-skill plan required")
    schedule = refinement_schedule(f["records"], cfg["seed"])
    if r["schedule_hash"] != str(hash_json(schedule)) or len(r["history"]) != 2048:
        raise ValueError("complete fixed actual2048-step history required")
    for i, (actual, indices) in enumerate(zip(r["history"], schedule, strict=True), start=1):
        if (
            actual["step"] != i
            or actual["indices"] != indices
            or any(
                type(actual[k]) not in (int, float) or not math.isfinite(actual[k]) or actual[k] < 0
                for k in ("loss", "old_loss", "current_loss", "teacher_loss", "gradient_norm")
            )
        ):
            raise ValueError("actual finite scheduled losses and gradient history required")
    acquired_model, _ = load_consolidated_adapter(
        parent,
        parent_lineage_hash=parent_lineage_hash,
        root=pp.parent,
        expected_receipt_hash=r["parent_training_hash"],
    )
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(cfg["seed"]).get_state())
        model = RefinementMotorAdapter(acquired_model)
    initial = {k: v.clone() for k, v in model.head.state_dict().items()}
    cp = Path(r["checkpoint_path"])
    if (
        cp.is_symlink()
        or cp.resolve().parent != root.resolve()
        or cp.stat().st_size > 16000000
        or digest(cp) != r["checkpoint_hash"]
    ):
        raise ValueError("confined bounded exact tensor-only refinement checkpoint required")
    state = torch.load(cp, map_location="cpu", weights_only=True)
    validate_refinement_checkpoint(
        state, dict(model.head.state_dict()), config=cfg, commitments={k: r[k] for k in COMMITMENTS}
    )
    if any(not torch.equal(v, state["initial_head"][k]) for k, v in initial.items()):
        raise ValueError("new head initialization does not match its claimed independent seed")
    model.head.load_state_dict(state["head"], strict=True)
    return model.eval().requires_grad_(False), r
