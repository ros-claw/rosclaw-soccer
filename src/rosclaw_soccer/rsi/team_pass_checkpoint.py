"""Strict separately named SIM admission of offline learned pass weights."""

import json
import math
import random
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
from rosclaw_soccer.rsi.team_pass_consolidation import BANK_STATUS, STATUS, pass_config, pass_pools


def load_team_pass(
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
        or rp.is_symlink()
        or dp.is_symlink()
        or (root / "failed.json").exists()
        or digest(rp) != expected_receipt_hash
    ):
        raise ValueError("exact completed offline pass adaptation required")
    r, d = [json.loads(x.read_text()) for x in (rp, dp)]
    cfg = pass_config(r["config"]["seed"], r["config"]["device"])
    if (
        r["status"] != STATUS
        or r["declaration_hash"] != digest(dp)
        or r["config"] != cfg
        or d["config"] != cfg
        or r["input_file_hashes"] != d["input_file_hashes"]
        or any(r[k] != d[k] for k in ("warm_start_hash", "data_hash", "parent_lineage_hash"))
        or r["parent_lineage_hash"] != parent_lineage_hash
        or r["offline_supervised_steps"] != 16
        or r["PPO_optimizer_steps_this_stage"] != 0
        or r["initial_optimizer_steps"] != 112
        or r["cumulative_optimizer_steps"] != 128
        or r["cumulative_critic_optimizer_steps"] != 48
        or r["physical_steps_executed"] != 0
        or r["online_RL_claim"] is not False
        or r["old_retention_proven"] is not False
        or len(r["history"]) != 16
        or len(r["cpu_replay"]) != 177
        or any(
            v[k] is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("separate fixed offline16/PPO0/critic0 pass phase accounting required")

    def receipt(h: str) -> tuple[Path, dict[str, Any]]:
        ps = [
            Path(p)
            for p, v in r["input_file_hashes"].items()
            if v == h and p.endswith("/complete.json")
        ]
        if len(ps) != 1 or ps[0].is_symlink() or digest(ps[0]) != h:
            raise ValueError("unique exact offline pass lineage required")
        return ps[0], json.loads(ps[0].read_text())

    wp, warm = receipt(r["warm_start_hash"])
    _, bank = receipt(r["data_hash"])
    if (
        bank["status"] != BANK_STATUS
        or bank["warm_start_hash"] != r["warm_start_hash"]
        or bank["parent_lineage_hash"] != parent_lineage_hash
        or warm["cumulative_optimizer_steps"] != 112
        or warm["cumulative_critic_optimizer_steps"] != 48
    ):
        raise ValueError("exact retained65023 actor112/critic48 and new audited bank required")
    old, teachers, new = pass_pools(bank["records"])
    model, _ = load_dual_candidate_for_sim_audit(
        parent,
        parent_lineage_hash=parent_lineage_hash,
        root=wp.parent,
        expected_receipt_hash=r["warm_start_hash"],
        checkpoint_kind="contact-consolidated-v2",
        _depth=depth + 1,
    )
    warm_path = Path(warm["checkpoint_path"])
    if digest(warm_path) != warm["checkpoint_hash"]:
        raise ValueError("unchanged validated warm network and optimizer required")
    warm_state = torch.load(warm_path, map_location="cpu", weights_only=True)
    paths = [Path(r[k + "_path"]) for k in ("initial_checkpoint", "checkpoint")]
    for path, key in zip(paths, ("initial_checkpoint_hash", "checkpoint_hash"), strict=True):
        if path.is_symlink() or path.parent.resolve() != root.resolve() or digest(path) != r[key]:
            raise ValueError("confined exact tensor-only pass checkpoints required")
    first, last = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    if (
        not same_training_state(first, warm_state)
        or last.get("schema") != "soccer.dual_motor.team_pass_consolidation.v1"
        or last.get("config") != cfg
        or last.get("optimizer_updates") != dict(actor=128, critic=48)
        or last.get("warm_start_hash") != r["warm_start_hash"]
        or last.get("parent_lineage_hash") != parent_lineage_hash
        or not same_training_state(last.get("critic"), first["critic"])
    ):
        raise ValueError(
            "exact inherited full training state and separate pass checkpoint required"
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
        raise ValueError("actual16 recurrent motor tensor changes required")
    parameters = list(model.plastic.parameters()) + list(MotorCritic().parameters())
    validate_adam_slots(first["optimizer"], parameters, 112, 48)
    validate_adam_slots(last["optimizer"], parameters, 128, 48)
    fo, io = last["optimizer"], first["optimizer"]
    if (
        fo["param_groups"][0]["lr"] != cfg["actor_lr"]
        or not same_training_state(fo["param_groups"][1], io["param_groups"][1])
        or any(
            not same_training_state(fo["state"][i], io["state"][i])
            for i in fo["param_groups"][1]["params"]
        )
    ):
        raise ValueError("unchanged persistent critic Adam and declared actor rate required")
    rng = random.Random(cfg["seed"])
    for i, entry in enumerate(r["history"], 1):
        teacher, new_anchor, anchors = rng.choice(teachers), rng.choice(new), rng.sample(old, 4)
        if (
            entry["step"] != i
            or entry["teacher_index"] != teacher
            or entry["new_anchor_index"] != new_anchor
            or entry["anchor_indices"] != anchors
            or any(
                type(entry[k]) not in (int, float) or not math.isfinite(entry[k])
                for k in ("loss", "imitation", "retention", "new_retention", "gradient_norm")
            )
        ):
            raise ValueError(
                "complete deterministic sample schedule and finite gradient history required"
            )
    model.plastic.load_state_dict(last["plastic_network"], strict=True)
    return model.eval().requires_grad_(False), r
