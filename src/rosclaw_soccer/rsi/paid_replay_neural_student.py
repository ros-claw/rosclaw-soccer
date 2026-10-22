"""Independent neural-fit verification and explicitly bounded native SIM loading."""

import json
from pathlib import Path
from typing import Any

import torch

from rosclaw_soccer.rsi.paid_canonical_behavior import digest
from rosclaw_soccer.rsi.paid_replay_neural import (
    FIT_STATUS,
    ReplayMotorHead,
    replay_certificate,
    replay_config,
    training_arrays,
)

AUDITED_STATUS = "PAID_SOFT_REPLAY_NEURAL_CPU_VERIFIED_NOT_PHYSICALLY_QUALIFIED"


def validate_neural_state(
    state: dict[str, Any], receipt: dict[str, Any], template: dict[str, torch.Tensor]
) -> None:
    cfg = replay_config(
        state.get("config", {}).get("seed_index"),
        tail_replay=state.get("config", {}).get("tail_replay", False),
    )
    if (
        state.get("schema") != "paid.soft_replay_neural.offline.v1"
        or receipt.get("status") != FIT_STATUS
        or receipt.get("config") != cfg
        or state.get("config") != cfg
        or receipt.get("optimizer_updates") != cfg["updates"]
        or receipt.get("own_native_steps") != 0
        or receipt.get("runtime_admitted") is not False
        or receipt.get("weights_only_checkpoint_reload") is not True
        or receipt.get("parent_parameters_exact_unchanged") is not True
        or any(receipt.get(k) is not False for k in ("hardware_authorized", "promotion_authorized"))
    ):
        raise ValueError("exact offline-only soft replay contract required")
    for k in (
        "config",
        "source_head",
        "data_hash",
        "body_scene_hash",
        "parent_hash",
        "parent_checkpoint_hash",
        "declaration_hash",
        "counts",
        "certificate",
    ):
        if k not in state or state[k] != receipt.get(k):
            raise ValueError("neural state/receipt identity differs")
    changed = []
    for name in ("model", "initial_model"):
        values = state.get(name, {})
        if not isinstance(values, dict) or set(values) != set(template):
            raise ValueError("exact normalization and eight network tensors required")
        for k, v in values.items():
            t = template[k]
            if (
                not isinstance(v, torch.Tensor)
                or v.shape != t.shape
                or v.dtype != t.dtype
                or v.device.type != "cpu"
                or v.layout != torch.strided
                or not torch.isfinite(v).all()
            ):
                raise ValueError("finite exact CPU neural tensors required")
            if name == "initial_model" and not torch.equal(v, t):
                raise ValueError("seeded neural initialization differs")
    for k, v in state["model"].items():
        if not torch.equal(v, template[k]):
            changed.append(k)
    params = {k: v for k, v in template.items() if k.startswith("net.")}
    counts = dict(
        completed_updates=cfg["updates"],
        nonzero_gradient_updates=cfg["updates"],
        active_forward_rows=cfg["updates"] * cfg["active_batch"],
        rehearsal_forward_rows=cfg["updates"] * cfg["rehearsal_batch"],
        changed_network_tensors=changed,
    )
    if cfg.get("tail_replay", False):
        refreshes = (cfg["updates"] - 1) // cfg["hard_refresh_every"] + 1
        counts.update(
            hard_pool_refreshes=refreshes,
            hard_pool_forward_rows=refreshes * receipt["certificate"]["rehearsal_rows"],
            uniform_rehearsal_forward_rows=cfg["updates"] * cfg["uniform_rehearsal_batch"],
            hard_rehearsal_forward_rows=cfg["updates"] * cfg["hard_rehearsal_batch"],
        )
    if set(changed) != set(params) or state["counts"] != counts:
        raise ValueError("actual effective neural updates and unchanged normalization required")
    optimizer = state.get("optimizer", {})
    groups, slots = optimizer.get("param_groups", []), optimizer.get("state", {})
    if (
        len(groups) != 1
        or groups[0].get("params") != list(range(len(params)))
        or groups[0].get("lr") != cfg["final_lr"]
        or groups[0].get("weight_decay") != 0
        or set(slots) != set(range(len(params)))
    ):
        raise ValueError("only declared neural parameters and final scheduler state required")
    for i, t in enumerate(params.values()):
        slot = slots[i]
        step = slot.get("step")
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or float(step) != cfg["updates"]:
            raise ValueError("actual completed Adam slot steps required")
        for k in ("exp_avg", "exp_avg_sq"):
            v = slot.get(k)
            if (
                not isinstance(v, torch.Tensor)
                or v.device.type != "cpu"
                or v.dtype != torch.float32
                or v.shape != t.shape
                or not torch.isfinite(v).all()
                or k == "exp_avg_sq"
                and (v < 0).any()
            ):
                raise ValueError("finite neural-only Adam moments required")


def load_replay_fit(
    root: Path, expected: str, policy: Path
) -> tuple[Any, ReplayMotorHead, dict[str, Any]]:
    from rosclaw_soccer.rsi.paid_growth_memory_data import load_growth_data

    rp, dp = root / "complete.json", root / "declaration.json"
    if (
        rp.is_symlink()
        or dp.is_symlink()
        or digest(rp) != expected
        or (root / "failed.json").exists()
    ):
        raise ValueError("closed exact soft-replay neural fit required")
    r, d = [json.loads(p.read_text()) for p in (rp, dp)]
    cp = Path(r["checkpoint_path"])
    if (
        digest(dp) != r["declaration_hash"]
        or cp.is_symlink()
        or cp.parent != root
        or cp.stat().st_size > 16000000
        or digest(cp) != r["checkpoint_hash"]
        or d.get("physical_rollout_budget") != 0
        or any(d.get(k) is not False for k in ("hardware_authorized", "promotion_authorized"))
    ):
        raise ValueError("confined bounded weights-only artifact and offline declaration required")
    cfg = replay_config(
        r["config"]["seed_index"], tail_replay=r["config"].get("tail_replay", False)
    )
    for k in (
        "config",
        "data_hash",
        "source_head",
        "body_scene_hash",
        "parent_hash",
        "parent_checkpoint_hash",
        "input_file_hashes",
    ):
        if r[k] != d[k]:
            raise ValueError("neural declaration identity differs")
    parent, data, z = load_growth_data(Path(d["data_root"]), r["data_hash"], policy)
    if any(data[k] != r[k] for k in ("body_scene_hash", "parent_hash", "parent_checkpoint_hash")):
        raise ValueError("exact reconstructed current motor parent required")
    x, _, _, _ = training_arrays(z)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg["seed"])
        head = ReplayMotorHead(
            x.double().mean(0).float(), x.double().std(0, unbiased=False).float().clamp_min(0.1)
        )
    state = torch.load(cp, map_location="cpu", weights_only=True)
    validate_neural_state(state, r, head.state_dict())
    head.load_state_dict(state["model"], strict=True)
    head.eval().requires_grad_(False)
    if replay_certificate(head, z) != r["certificate"]:
        raise ValueError("independently recomputed all-row CPU batch1 certificate differs")
    return parent, head, r


class FrozenReplayResidual(torch.nn.Module):
    """SIM-only research motor, with unchanged final R2-relative authority."""

    bound_radians = 0.2

    def __init__(self, parent: torch.nn.Module, head: ReplayMotorHead) -> None:
        from rosclaw_soccer.rsi.paid_cardinal_memory import FrozenCardinalResidual

        super().__init__()
        if type(parent) is not FrozenCardinalResidual or type(head) is not ReplayMotorHead:
            raise ValueError("exact current canonical parent and neural replay head required")
        self.parent = parent.eval().requires_grad_(False)
        self.head = head.eval().requires_grad_(False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        from rosclaw_soccer.rsi.paid_compact_support import check_features

        check_features(features)
        return self.head.combine(features, self.parent(features))


def load_replay_student(
    policy: Path,
    *,
    checkpoint: Path,
    checkpoint_hash: str,
    training_receipt: Path,
    training_receipt_hash: str,
    scene_hash: str,
) -> tuple[Any, dict[str, Any]]:
    """Permit native SIM research only from independently reconstructed fits."""
    from rosclaw_soccer.rsi.paid_full_entry_student import FrozenFullEntryActor

    if (
        training_receipt.is_symlink()
        or digest(training_receipt) != training_receipt_hash
        or (training_receipt.parent / "failed.json").exists()
    ):
        raise ValueError("exact completed independent neural audit required")
    audit = json.loads(training_receipt.read_text())
    if (
        audit.get("status") != AUDITED_STATUS
        or audit.get("CPU_fit_reconstruction") is not True
        or audit.get("runtime_admitted") is not False
        or audit.get("own_native_steps") != 0
        or audit.get("optimizer_updates") != 0
        or any(audit.get(k) is not False for k in ("hardware_authorized", "promotion_authorized"))
        or audit.get("body_scene_hash") != scene_hash
        or audit.get("checkpoint_path") != str(checkpoint)
        or audit.get("checkpoint_hash") != checkpoint_hash
    ):
        raise ValueError("CPU-verified same-body SIM candidate required, not fit-only admission")
    parent, head, fitted = load_replay_fit(Path(audit["fit_root"]), audit["fit_hash"], policy)
    if any(
        audit.get(k) != fitted.get(k)
        for k in (
            "config",
            "certificate",
            "counts",
            "checkpoint_path",
            "checkpoint_hash",
            "body_scene_hash",
            "parent_hash",
        )
    ):
        raise ValueError("independent fit/body/normalization/moment certificate changed")
    residual = FrozenReplayResidual(parent.residual, head).eval()
    return FrozenFullEntryActor(
        parent.base, residual, parent.get_buffer("action_scale")
    ).eval(), audit
