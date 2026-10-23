"""Strict SIM admission of actual own-state pass correction weights, phase v2.

Carries the full128/48 training state, admits exactly64 additional BC actor
updates and no critic/PPO update. Never authorizes a policy promotion or REAL.
"""

import json
import math
from pathlib import Path
from typing import Any

import torch

from rosclaw_soccer.rsi.dual_motor_checkpoint import digest, validate_network
from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import (
    load_dual_candidate_for_sim_audit,
    same_training_state,
    validate_adam_slots,
)
from rosclaw_soccer.rsi.dual_recurrent_actor_critic import MotorCritic
from rosclaw_soccer.rsi.dual_recurrent_motor import DualRecurrentMotor
from rosclaw_soccer.rsi.paid_full_entry_student import FrozenFullEntryActor
from rosclaw_soccer.rsi.team_pass_correction import (
    CORRECTION_BANK_STATUS,
    CORRECTION_KIND,
    CORRECTION_STATUS,
    correction_config,
    correction_schedule,
)


def load_team_pass_correction(
    parent: FrozenFullEntryActor,
    *,
    parent_lineage_hash: str,
    root: Path,
    expected_receipt_hash: str,
    depth: int,
) -> tuple[DualRecurrentMotor, dict[str, Any]]:
    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        root.is_symlink()
        or any(x.is_symlink() for x in (rp, dp))
        or (root / "failed.json").exists()
        or digest(rp) != expected_receipt_hash
    ):
        raise ValueError("exact complete independently versioned correction phase required")
    r, d = [json.loads(x.read_text()) for x in (rp, dp)]
    cfg = correction_config(r["config"]["seed"], r["config"]["device"])
    if (
        r["status"] != CORRECTION_STATUS
        or r["config"] != cfg
        or d["config"] != cfg
        or r["declaration_hash"] != digest(dp)
        or r["input_file_hashes"] != d["input_file_hashes"]
        or r["parent_lineage_hash"] != parent_lineage_hash
        or any(
            r[k] != d[k]
            for k in ("warm_start_hash", "data_hash", "parent_lineage_hash", "source_head")
        )
        or r["offline_supervised_steps"] != 64
        or len(r["history"]) != 64
        or r["initial_optimizer_steps"] != 128
        or r["cumulative_optimizer_steps"] != 192
        or r["initial_critic_optimizer_steps"] != 48
        or r["cumulative_critic_optimizer_steps"] != 48
        or r["PPO_optimizer_steps_this_stage"] != 0
        or r["cumulative_PPO_actor_steps"] != 16
        or r["physical_steps_executed"] != 0
        or len(r["cpu_replay"]) != 274
        or any(r[k] is not False for k in ("online_RL_claim", "old_retention_proven"))
        or any(
            v[k] is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("exact offline64/actor128to192/critic48/PPO0 correction phase required")

    def receipt(h: str) -> tuple[Path, dict[str, Any]]:
        paths = [
            Path(p)
            for p, v in r["input_file_hashes"].items()
            if v == h and p.endswith("/complete.json")
        ]
        if len(paths) != 1 or paths[0].is_symlink() or digest(paths[0]) != h:
            raise ValueError("unique exact correction lineage required")
        return paths[0], json.loads(paths[0].read_text())

    wp, warm = receipt(r["warm_start_hash"])
    _, bank = receipt(r["data_hash"])
    if (
        bank["status"] != CORRECTION_BANK_STATUS
        or bank["warm_start_hash"] != r["warm_start_hash"]
        or bank["parent_lineage_hash"] != parent_lineage_hash
        or warm["config"]["seed"] != 75024
        or warm["cumulative_optimizer_steps"] != 128
        or warm["cumulative_critic_optimizer_steps"] != 48
    ):
        raise ValueError("same retained research student and whole actual correction bank required")
    schedule = correction_schedule(bank["records"], cfg["seed"])
    for entry, expected in zip(r["history"], schedule, strict=True):
        if any(entry[k] != v for k, v in expected.items()) or any(
            type(entry[k]) not in (int, float) or not math.isfinite(entry[k])
            for k in ("loss", "imitation", "retention", "new_retention", "gradient_norm")
        ):
            raise ValueError(
                "whole deterministic fully covering schedule and finite history required"
            )
    model, _ = load_dual_candidate_for_sim_audit(
        parent,
        parent_lineage_hash=parent_lineage_hash,
        root=wp.parent,
        expected_receipt_hash=r["warm_start_hash"],
        checkpoint_kind="team-pass-consolidated-v1",
        _depth=depth + 1,
    )
    warm_path = Path(warm["checkpoint_path"])
    if warm_path.is_symlink() or digest(warm_path) != warm["checkpoint_hash"]:
        raise ValueError("exact inherited full Adam and network checkpoint required")
    inherited = torch.load(warm_path, map_location="cpu", weights_only=True)
    paths = [Path(r[k + "_path"]) for k in ("initial_checkpoint", "checkpoint")]
    for path, key in zip(paths, ("initial_checkpoint_hash", "checkpoint_hash"), strict=True):
        if path.is_symlink() or path.parent.resolve() != root.resolve() or digest(path) != r[key]:
            raise ValueError("confined exact safe tensor-only correction checkpoints required")
    first, last = [torch.load(x, map_location="cpu", weights_only=True) for x in paths]
    if (
        not same_training_state(first, inherited)
        or last.get("schema") != "soccer.dual_motor.team_pass_correction.v2"
        or last.get("config") != cfg
        or last.get("optimizer_updates") != dict(actor=192, critic=48)
        or last.get("warm_start_hash") != r["warm_start_hash"]
        or last.get("parent_lineage_hash") != parent_lineage_hash
        or not same_training_state(last.get("critic"), first["critic"])
    ):
        raise ValueError(
            "complete inherited training state and exact new phase checkpoint required"
        )
    validate_network(last.get("plastic_network"), model.plastic.state_dict())
    changed = [
        k
        for k, v in last["plastic_network"].items()
        if not torch.equal(v, first["plastic_network"][k])
    ]
    if (
        set(changed) != set(dict(model.plastic.named_parameters()))
        or changed != r["changed_parameter_tensors"]
    ):
        raise ValueError("all and only16 plastic parameters must actually change")
    parameters = list(model.plastic.parameters()) + list(MotorCritic().parameters())
    validate_adam_slots(first["optimizer"], parameters, 128, 48)
    validate_adam_slots(last["optimizer"], parameters, 192, 48, actor_phase=CORRECTION_KIND)
    fo, io = last["optimizer"], first["optimizer"]
    if (
        fo["param_groups"][0]["lr"] != cfg["actor_lr"]
        or not same_training_state(fo["param_groups"][1], io["param_groups"][1])
        or any(
            not same_training_state(fo["state"][i], io["state"][i])
            for i in fo["param_groups"][1]["params"]
        )
    ):
        raise ValueError("critic weights/Adam slots/group must remain exactly frozen")
    model.plastic.load_state_dict(last["plastic_network"], strict=True)
    return model.eval().requires_grad_(False), r
