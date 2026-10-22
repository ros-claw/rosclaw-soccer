"""Strict SIM admission of offline consolidation; never reinterpret it as PPO."""

import json
import random
from pathlib import Path
from typing import Any

import torch

from rosclaw_soccer.rsi.contact_consolidation import STATUS, consolidation_config
from rosclaw_soccer.rsi.contact_consolidation_data import BANK_STATUS
from rosclaw_soccer.rsi.dual_motor_checkpoint import digest, validate_network
from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import (
    load_dual_candidate_for_sim_audit,
    same_training_state,
    validate_adam_slots,
)
from rosclaw_soccer.rsi.dual_recurrent_actor_critic import MotorCritic
from rosclaw_soccer.rsi.dual_recurrent_motor import DualRecurrentMotor
from rosclaw_soccer.rsi.paid_full_entry_student import FrozenFullEntryActor


def load_contact_consolidated(
    parent: FrozenFullEntryActor,
    *,
    parent_lineage_hash: str,
    root: Path,
    expected_receipt_hash: str,
    depth: int,
) -> tuple[DualRecurrentMotor, dict[str, Any]]:
    rp, dp = root / "complete.json", root / "declaration.json"
    if root.is_symlink() or (root / "failed.json").exists() or digest(rp) != expected_receipt_hash:
        raise ValueError("exact closed offline consolidation receipt required")
    r, d = [json.loads(x.read_text()) for x in (rp, dp)]
    if (
        r["status"] != STATUS
        or r["declaration_hash"] != digest(dp)
        or r["config"] != d["config"]
        or r["config"] != consolidation_config(r["config"]["seed"], r["config"]["device"])
        or r["input_file_hashes"] != d["input_file_hashes"]
        or any(r[k] != d[k] for k in ("warm_start_hash", "data_hash", "parent_lineage_hash"))
        or r["parent_lineage_hash"] != parent_lineage_hash
        or any(
            v[k] is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
        or r["online_RL_claim"] is not False
        or r["old_retention_proven"] is not False
        or r["offline_supervised_steps"] != 64
        or r["PPO_optimizer_steps_this_stage"] != 0
        or r["cumulative_PPO_actor_steps"] != 16
        or r["initial_optimizer_steps"] != 16
        or r["initial_critic_optimizer_steps"] != 48
        or r["cumulative_optimizer_steps"] != 80
        or r["cumulative_critic_optimizer_steps"] != 48
        or r["physical_steps_executed"] != 0
        or r["new_rollout_generations"] != 0
        or len(r["history"]) != 64
        or len(r["cpu_replay"]) != 65
    ):
        raise ValueError("separate exact offline64/PPO0/critic0 phase accounting required")

    def receipt(h: str) -> tuple[Path, dict[str, Any]]:
        paths = [
            Path(x)
            for x, v in r["input_file_hashes"].items()
            if v == h and x.endswith("/complete.json")
        ]
        if len(paths) != 1 or digest(paths[0]) != h:
            raise ValueError("unique exact consolidation lineage required")
        return paths[0], json.loads(paths[0].read_text())

    wp, warm = receipt(r["warm_start_hash"])
    _, data = receipt(r["data_hash"])
    if (
        data["status"] != BANK_STATUS
        or data["warm_start_hash"] != r["warm_start_hash"]
        or data["parent_lineage_hash"] != parent_lineage_hash
        or len(data["records"]) != 65
        or warm["cumulative_optimizer_steps"] != 16
        or warm["cumulative_critic_optimizer_steps"] != 48
    ):
        raise ValueError("exact qualified bank65 and W6 actor16/critic48 required")
    model, _ = load_dual_candidate_for_sim_audit(
        parent,
        parent_lineage_hash=parent_lineage_hash,
        root=wp.parent,
        expected_receipt_hash=r["warm_start_hash"],
        checkpoint_kind="ar1-actor-critic",
        _depth=depth + 1,
    )
    warm_path = Path(warm["checkpoint_path"])
    if digest(warm_path) != warm["checkpoint_hash"]:
        raise ValueError("exact safe warm checkpoint required")
    warm_state = torch.load(warm_path, map_location="cpu", weights_only=True)
    paths = [Path(r[k + "_path"]) for k in ("initial_checkpoint", "checkpoint")]
    for path, key in zip(paths, ("initial_checkpoint_hash", "checkpoint_hash"), strict=True):
        if path.is_symlink() or path.parent.resolve() != root.resolve() or digest(path) != r[key]:
            raise ValueError("confined exact safe consolidation checkpoint required")
    first, last = [torch.load(x, map_location="cpu", weights_only=True) for x in paths]
    if not same_training_state(first, warm_state):
        raise ValueError("initial checkpoint must carry ALL exact warm state before BC schedule")
    if (
        last.get("schema") != "soccer.dual_motor.contact_consolidation.v1"
        or last.get("config") != r["config"]
        or last.get("optimizer_updates") != dict(actor=80, critic=48)
        or last.get("warm_start_hash") != r["warm_start_hash"]
        or last.get("parent_lineage_hash") != parent_lineage_hash
        or not same_training_state(last.get("critic"), first["critic"])
    ):
        raise ValueError("separate consolidated schema and unchanged critic required")
    validate_network(last.get("plastic_network"), model.plastic.state_dict())
    changed = [
        k
        for k, v in last["plastic_network"].items()
        if not torch.equal(v, first["plastic_network"][k])
    ]
    if len(changed) != 16 or changed != r["changed_parameter_tensors"]:
        raise ValueError("actual16 recurrent parameter changes required")
    critic = MotorCritic()
    parameters = list(model.plastic.parameters()) + list(critic.parameters())
    validate_adam_slots(first["optimizer"], parameters, 16, 48)
    validate_adam_slots(last["optimizer"], parameters, 80, 48)
    final_opt, initial_opt = last["optimizer"], first["optimizer"]
    if (
        final_opt["param_groups"][0]["lr"] != 0.0001
        or not same_training_state(final_opt["param_groups"][1], initial_opt["param_groups"][1])
        or any(
            not same_training_state(final_opt["state"][i], initial_opt["state"][i])
            for i in final_opt["param_groups"][1]["params"]
        )
    ):
        raise ValueError("declared BC actor rate and exact frozen critic Adam required")
    old = [i for i, x in enumerate(data["records"]) if x["role"] == "retained_W6_skill"]
    teachers = [
        i
        for i, x in enumerate(data["records"])
        if x["bc_weight"] == 1 and x["role"] != "retained_W6_skill"
    ]
    if len(old) != 14 or len(teachers) != 48:
        raise ValueError("ALL14 anchors and ALL48 positive teachers required")
    rng = random.Random(r["config"]["seed"])
    for i, entry in enumerate(r["history"], 1):
        teacher, anchors = rng.choice(teachers), rng.sample(old, 4)
        if (
            entry["step"] != i
            or entry["teacher_index"] != teacher
            or entry["anchor_indices"] != anchors
            or any(
                not isinstance(entry[k], (int, float)) or not torch.isfinite(torch.tensor(entry[k]))
                for k in ("loss", "imitation", "retention", "gradient_norm")
            )
        ):
            raise ValueError("exact seed-bound complete whole-history sample schedule required")
    model.plastic.load_state_dict(last["plastic_network"], strict=True)
    return model.eval().requires_grad_(False), r
