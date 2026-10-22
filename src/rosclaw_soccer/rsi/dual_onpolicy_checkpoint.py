"""Separate safe SIM research admission for an actual AR actor-critic checkpoint.

Never reinterpret PPO steps as the supervised loader's 256 Adam updates.
Optimizer metadata is validated, not an independent gradient reexecution.
"""

import json
from pathlib import Path
from typing import Any

import torch

from rosclaw_soccer.rsi.dual_motor_checkpoint import (
    digest,
    load_dual_for_sim_audit,
    validate_network,
)
from rosclaw_soccer.rsi.dual_recurrent_actor_critic import MotorCritic
from rosclaw_soccer.rsi.dual_recurrent_motor import DualRecurrentMotor
from rosclaw_soccer.rsi.paid_full_entry_student import FrozenFullEntryActor


def validate_adam_slots(opt: Any, parameters: list[torch.Tensor], steps: int) -> None:
    if type(steps) is not int or not 1 <= steps <= 4 or type(opt) is not dict:
        raise ValueError("bounded accepted actual PPO optimizer steps required")
    groups, slots = opt.get("param_groups", []), opt.get("state", {})
    if (
        len(groups) != 2
        or [len(g.get("params", [])) for g in groups] != [16, 4]
        or len(parameters) != 20
    ):
        raise ValueError("exact16actor+4critic parameter ownership required")
    ids = [i for g in groups for i in g["params"]]
    if len(set(ids)) != 20 or set(ids) != set(slots):
        raise ValueError("all and only20 unique actual Adam slots required")
    for i, p in zip(ids, parameters, strict=True):
        s = slots[i]
        if set(s) != {"step", "exp_avg", "exp_avg_sq"} or float(s["step"]) != steps:
            raise ValueError("accepted actual PPO steps must bind every optimizer slot")
        for name in ("exp_avg", "exp_avg_sq"):
            x = s[name]
            if (
                not isinstance(x, torch.Tensor)
                or x.device.type != "cpu"
                or x.dtype != torch.float32
                or x.shape != p.shape
                or not torch.isfinite(x).all()
            ):
                raise ValueError("finite exact-shape CPU PPO optimizer moments required")
        if (s["exp_avg_sq"] < 0).any():
            raise ValueError("negative Adam second moment")


def validate_critic(network: Any, expected: dict[str, torch.Tensor]) -> None:
    if type(network) is not dict or set(network) != set(expected):
        raise ValueError("exact tensor-only critic network required")
    for k, x in network.items():
        if (
            not isinstance(x, torch.Tensor)
            or x.device.type != "cpu"
            or x.dtype != torch.float32
            or x.shape != expected[k].shape
            or not torch.isfinite(x).all()
        ):
            raise ValueError("finite exact CPU critic weights required")


def load_dual_candidate_for_sim_audit(
    parent: FrozenFullEntryActor,
    *,
    parent_lineage_hash: str,
    root: Path,
    expected_receipt_hash: str,
    checkpoint_kind: str = "supervised",
) -> tuple[DualRecurrentMotor, dict[str, Any]]:
    if checkpoint_kind == "supervised":
        return load_dual_for_sim_audit(
            parent,
            parent_lineage_hash=parent_lineage_hash,
            root=root,
            expected_receipt_hash=expected_receipt_hash,
        )
    if checkpoint_kind != "ar1-actor-critic":
        raise ValueError("explicit research checkpoint kind required")
    rp, dp = root / "complete.json", root / "declaration.json"
    if root.is_symlink() or (root / "failed.json").exists() or digest(rp) != expected_receipt_hash:
        raise ValueError("exact closed actor-critic training receipt required")
    r, d = [json.loads(x.read_text()) for x in (rp, dp)]
    if (
        r["status"] != "DUAL_AR1_ACTOR_CRITIC_ACTUALLY_UPDATED_PENDING_PHYSICAL_EXAM"
        or r["declaration_hash"] != digest(dp)
        or r["config"] != d["config"]
        or r["config"]["protocol"] != "dual256.AR1_audited_option_actor_critic.v1"
        or r["input_file_hashes"] != d["input_file_hashes"]
        or r["parent_lineage_hash"] != parent_lineage_hash
        or d["parent_lineage_hash"] != parent_lineage_hash
        or r["warm_start_hash"] != d["warm_start_hash"]
        or r["data_hash"] != d["data_hash"]
        or r["experience_audit_hash"] != d["experience_audit_hash"]
        or r["experience_hash"] != d["experience_hash"]
        or r["online_RL_claim"] is not True
        or r["old_retention_proven"] is not False
        or r["physical_steps_executed"] != 0
        or r["new_rollout_generations"] != 0
        or len(r["epochs"]) != 4
        or any(
            v[k] is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("fixed audited pure-learner actor-critic declaration required")
    steps = r["accepted_optimizer_steps"]
    if steps != sum(int(x["decision"]["applied"]) for x in r["epochs"]):
        raise ValueError("truthful accepted/rejected optimizer accounting required")
    paths = [
        Path(x)
        for x, h in r["input_file_hashes"].items()
        if h == r["warm_start_hash"] and x.endswith("/complete.json")
    ]
    if len(paths) != 1:
        raise ValueError("unique exact learned on-policy predecessor required")
    model, _ = load_dual_for_sim_audit(
        parent,
        parent_lineage_hash=parent_lineage_hash,
        root=paths[0].parent,
        expected_receipt_hash=r["warm_start_hash"],
    )
    first_path, last_path = Path(r["initial_checkpoint_path"]), Path(r["checkpoint_path"])
    for x, h in ((first_path, r["initial_checkpoint_hash"]), (last_path, r["checkpoint_hash"])):
        if x.parent.resolve() != root.resolve() or digest(x) != h:
            raise ValueError("confined exact safe actor-critic checkpoint required")
    first, last = [
        torch.load(x, map_location="cpu", weights_only=True) for x in (first_path, last_path)
    ]
    if (
        any(
            type(x) is not dict
            or x.get("schema") != "soccer.dual_motor.ar1_ppo.v1"
            or x.get("warm_start_hash") != r["warm_start_hash"]
            or x.get("parent_lineage_hash") != parent_lineage_hash
            for x in (first, last)
        )
        or first.get("optimizer_updates") != 0
        or last.get("optimizer_updates") != steps
        or last.get("config") != r["config"]
    ):
        raise ValueError("separate exact initial/final PPO checkpoint schema required")
    expected = model.plastic.state_dict()
    for x in (first, last):
        validate_network(x.get("plastic_network"), expected)
    if any(not torch.equal(v, expected[k]) for k, v in first["plastic_network"].items()):
        raise ValueError("PPO must start at exact actual behavior-policy weights")
    changed = [
        k
        for k, v in last["plastic_network"].items()
        if not torch.equal(v, first["plastic_network"][k])
    ]
    if len(changed) != 16 or changed != r["changed_parameter_tensors"]:
        raise ValueError("actual end-to-end recurrent motor change required")
    critic = MotorCritic()
    for x in (first, last):
        validate_critic(x.get("critic"), critic.state_dict())
    if (
        first["critic"]["net.2.weight"].count_nonzero()
        or first["critic"]["net.2.bias"].count_nonzero()
    ):
        raise ValueError("declared initial zero baseline must really be zero-output")
    changed_critic = [
        k for k, v in last["critic"].items() if not torch.equal(v, first["critic"][k])
    ]
    if not changed_critic or changed_critic != r["changed_critic_tensors"]:
        raise ValueError("actual learned critic change required")
    validate_adam_slots(
        last["optimizer"], list(model.plastic.parameters()) + list(critic.parameters()), steps
    )
    model.plastic.load_state_dict(last["plastic_network"], strict=True)
    model.eval().requires_grad_(False)
    return model, r
