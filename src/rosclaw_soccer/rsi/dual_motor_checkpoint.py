"""Strict SIM research loader, not a runtime or hardware policy admission."""

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from rosclaw_soccer.rsi.dual_recurrent_motor import DualRecurrentMotor
from rosclaw_soccer.rsi.paid_full_entry_student import FrozenFullEntryActor


def digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("bounded ordinary research checkpoint/receipt required")
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def validate_network(network: Any, expected: dict[str, torch.Tensor]) -> None:
    if type(network) is not dict or set(network) != set(expected):
        raise ValueError("exact dual plastic-network tensor keys required")
    for k, v in network.items():
        if (
            not isinstance(v, torch.Tensor)
            or v.device.type != "cpu"
            or v.dtype != torch.float32
            or v.shape != expected[k].shape
            or not torch.isfinite(v).all()
        ):
            raise ValueError("finite exact CPU float32 plastic tensor required: " + k)
    if any(not torch.equal(network[k], expected[k]) for k in ("mean", "denominator")):
        raise ValueError("original qualified normalization must remain immutable")


def load_dual_for_sim_audit(
    parent: FrozenFullEntryActor,
    *,
    parent_lineage_hash: str,
    root: Path,
    expected_receipt_hash: str,
) -> tuple[DualRecurrentMotor, dict[str, Any]]:
    rp, dp = root / "complete.json", root / "declaration.json"
    if root.is_symlink() or (root / "failed.json").exists() or digest(rp) != expected_receipt_hash:
        raise ValueError("exact closed research training receipt required")
    r, d = [json.loads(p.read_text()) for p in (rp, dp)]
    if (
        r["status"] != "DUAL_RECURRENT_MOTOR_ACTUALLY_TRAINED_NOT_PHYSICALLY_QUALIFIED"
        or digest(dp) != r["declaration_hash"]
        or r["parent_lineage_hash"] != parent_lineage_hash
        or d["parent_lineage_hash"] != parent_lineage_hash
        or r["config"] != d["config"]
        or r["input_file_hashes"] != d["input_file_hashes"]
        or r["optimizer_updates"] != 256
        or r["physical_steps_executed"] != 0
        or r["online_RL_claim"] is not False
        or r["old_retention_proven"] is not False
        or any(
            v[k] is not False
            for v in (r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("fixed unpromoted full-history supervised training declaration required")
    initial_path, final_path = Path(r["initial_checkpoint_path"]), Path(r["checkpoint_path"])
    for p, h in ((initial_path, r["initial_checkpoint_hash"]), (final_path, r["checkpoint_hash"])):
        if p.parent.resolve() != root.resolve() or digest(p) != h:
            raise ValueError("confined exact source-bound safe checkpoint required")
    first, last = [
        torch.load(p, map_location="cpu", weights_only=True) for p in (initial_path, final_path)
    ]
    if (
        type(first) is not dict
        or type(last) is not dict
        or first.get("schema") != "soccer.dual_motor.plastic.v1"
        or last.get("schema") != first["schema"]
        or first.get("optimizer_updates") != 0
        or last.get("optimizer_updates") != 256
        or last.get("hidden_width") != 256
        or last.get("config") != r["config"]
        or any(
            v.get("data_hash") != r["data_hash"]
            or v.get("parent_lineage_hash") != parent_lineage_hash
            for v in (first, last)
        )
    ):
        raise ValueError("exact initial/final safe dual256 checkpoint metadata required")
    model = DualRecurrentMotor(parent).eval()
    expected = model.plastic.state_dict()
    validate_network(first.get("plastic_network"), expected)
    if any(not torch.equal(v, expected[k]) for k, v in first["plastic_network"].items()):
        raise ValueError(
            "warm start must equal actual qualified inherited network, not random weights"
        )
    validate_network(last.get("plastic_network"), expected)
    changed = [
        k
        for k, v in last["plastic_network"].items()
        if not torch.equal(v, first["plastic_network"][k])
    ]
    if len(changed) != 16 or changed != r["changed_parameter_tensors"]:
        raise ValueError("actual recurrent and action-network parameter changes must bind")
    opt = last.get("optimizer", {})
    groups = opt.get("param_groups", [])
    slots = opt.get("state", {})
    parameters = list(model.plastic.parameters())
    if (
        len(groups) != 1
        or len(groups[0].get("params", [])) != 16
        or len(slots) != 16
        or len(parameters) != 16
    ):
        raise ValueError("all and only16 actual Adam slots required")
    ids = groups[0]["params"]
    if len(set(ids)) != 16 or set(ids) != set(slots):
        raise ValueError("exact unique Adam parameter IDs required")
    for pid, p in zip(ids, parameters, strict=True):
        moment = slots[pid]
        if float(moment["step"]) != 256:
            raise ValueError("actual optimizer step counters must match all256 updates")
        for k in ("exp_avg", "exp_avg_sq"):
            v = moment[k]
            if (
                not isinstance(v, torch.Tensor)
                or v.device.type != "cpu"
                or v.dtype != torch.float32
                or v.shape != p.shape
                or not torch.isfinite(v).all()
            ):
                raise ValueError("finite exact optimizer moments required")
        if (moment["exp_avg_sq"] < 0).any():
            raise ValueError("invalid Adam second moment")
    model.plastic.load_state_dict(last["plastic_network"], strict=True)
    model.eval().requires_grad_(False)
    return model, r
