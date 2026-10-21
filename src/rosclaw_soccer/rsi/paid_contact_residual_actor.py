"""Bounded typed admission for a separate frozen-base research residual."""

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from rosclaw_soccer.rsi.paid_contact_residual import (
    ContactResidual,
    FrozenContactResidualActor,
    residual_config,
)
from rosclaw_soccer.rsi.paid_live_distilled_actor import load_live_distilled_actor
from rosclaw_soccer.rsi.paid_teacher_policy import PaidResearchPolicy
from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH


def validate_contact_state(
    state: dict[str, Any],
    receipt: dict[str, Any],
    template: dict[str, torch.Tensor],
    *,
    declaration_hash: str,
    scene_hash: str,
) -> None:
    config = residual_config(state.get("config", {}).get("variant"))
    counts = dict(rescue_forward_rows=377344, zero_forward_rows=598016, optimizer_updates=1024)
    if (
        state.get("schema") != "soccer.paid_contact_residual.v1"
        or state.get("epoch") != 256
        or state.get("config") != config
        or receipt.get("config") != config
        or state.get("counts") != counts
        or receipt.get("counts") != counts
        or state.get("declaration_hash") != declaration_hash
        or state.get("body_scene_hash") != scene_hash
        or state.get("original_actor_hash") != PUBLISHED_HASH
        or state.get("source_head") != receipt.get("source_head")
        or receipt.get("status") != "PAID_CONTACT_RESIDUAL_TRAINED_NOT_PHYSICALLY_EXAMINED"
        or receipt.get("native_steps") != 0
        or receipt.get("CPU_zero_neural_replay_rows") != 7620
        or receipt.get("CPU_feature_recompute_rows") != 3810
        or any(
            x.get(k) is not False
            for x in (state, receipt)
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "commercial_publicity_authorized",
            )
        )
    ):
        raise ValueError("exact completed residual training protocol and research ceiling required")
    network = state.get("residual", {})
    if not isinstance(network, dict) or set(network) != set(template):
        raise ValueError("exact separate contact/gate network keys required")
    for key, expected in template.items():
        value = network[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or value.layout != torch.strided
            or value.shape != expected.shape
            or not torch.isfinite(value).all()
        ):
            raise ValueError("finite exact CPU residual tensor required: " + key)
    opt = state.get("optimizer", {})
    groups = opt.get("param_groups", [])
    names = list(template)
    if (
        len(names) != 10
        or len(groups) != 1
        or groups[0].get("params") != list(range(10))
        or groups[0].get("lr") != config["lr"]
        or set(opt.get("state", {})) != set(range(10))
    ):
        raise ValueError("optimizer may contain only ten residual/gate tensors, never base weights")
    for pid, key in enumerate(names):
        slot = opt["state"][pid]
        step = slot.get("step")
        if (
            not isinstance(step, torch.Tensor)
            or step.device.type != "cpu"
            or step.numel() != 1
            or float(step) != 1024
        ):
            raise ValueError("actual complete residual Adam step count required")
        for field in ("exp_avg", "exp_avg_sq"):
            value = slot.get(field)
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float32
                or value.layout != torch.strided
                or value.shape != template[key].shape
                or not torch.isfinite(value).all()
                or (field == "exp_avg_sq" and (value < 0).any())
            ):
                raise ValueError("finite shape-bound residual Adam moments required")


def load_contact_residual_actor(
    policy: Path,
    *,
    checkpoint: Path,
    checkpoint_hash: str,
    training_receipt: Path,
    training_receipt_hash: str,
    scene_hash: str,
) -> tuple[FrozenContactResidualActor, dict[str, Any]]:
    def digest(path: Path) -> str:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("bounded ordinary residual checkpoint/receipt required")
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    if digest(checkpoint) != checkpoint_hash or digest(training_receipt) != training_receipt_hash:
        raise ValueError("exact completed residual weights and receipt required")
    receipt = json.loads(training_receipt.read_text())
    if receipt.get("final_checkpoint_hash") != checkpoint_hash:
        raise ValueError("only predeclared final residual weights may enter exam")
    dp = training_receipt.parent / "declaration.json"
    dh = digest(dp)
    declaration = json.loads(dp.read_text())
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("safe typed residual checkpoint required")
    residual = ContactResidual()
    validate_contact_state(
        state, receipt, residual.state_dict(), declaration_hash=dh, scene_hash=scene_hash
    )
    pins = receipt["input_file_hashes"]
    if (
        declaration.get("input_file_hashes") != pins
        or declaration.get("config") != receipt["config"]
        or declaration.get("base_frozen") is not True
        or declaration.get("teacher_hidden_states_copied_as_inputs") is not False
        or declaration.get("runtime_features") != "ownnormalized160+ownfrozenrecurrent128"
        or declaration.get("CPU_zero_neural_replay_rows") != 7620
        or declaration.get("CPU_feature_recompute_rows") != 3810
        or [w["case"] for w in declaration.get("worlds", [])] != list(range(16))
        or declaration.get("body_scene_hash") != scene_hash
        or declaration.get("original_actor_hash") != PUBLISHED_HASH
    ):
        raise ValueError("complete physical zero/curation/own-feature declaration required")

    def unique(h: str) -> Path:
        paths = [Path(p) for p, ph in pins.items() if ph == h]
        if len(paths) != 1 or digest(paths[0]) != h:
            raise ValueError("one exact source-bound motor or proof required")
        return paths[0]

    for key in ("zero_proof_hash", "zero_audit_hash", "curation_hash", "curation_audit_hash"):
        unique(declaration[key])
    if any(
        state.get(k) != declaration.get(k)
        for k in ("base_checkpoint_hash", "base_training_receipt_hash")
    ):
        raise ValueError("runtime base must be exact immutable training motor")
    base, _ = load_live_distilled_actor(
        policy,
        checkpoint=unique(state["base_checkpoint_hash"]),
        checkpoint_hash=state["base_checkpoint_hash"],
        training_receipt=unique(state["base_training_receipt_hash"]),
        training_receipt_hash=state["base_training_receipt_hash"],
        scene_hash=scene_hash,
    )
    initial_path = training_receipt.parent / "contact-residual-epoch0.pt"
    if digest(initial_path) != receipt.get("initial_checkpoint_hash"):
        raise ValueError("actual zero residual warmstart checkpoint differs")
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    if (
        initial.get("epoch") != 0
        or initial.get("schema") != state["schema"]
        or initial.get("declaration_hash") != dh
        or initial.get("base_checkpoint_hash") != state["base_checkpoint_hash"]
        or set(initial.get("residual", {})) != set(residual.state_dict())
    ):
        raise ValueError("exact zero-start residual declaration required")
    for key, value in initial["residual"].items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != residual.state_dict()[key].shape
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or not torch.isfinite(value).all()
        ):
            raise ValueError("finite exact initial residual tensors required")
    if (
        any(initial["residual"][k].any() for k in ("net.4.weight", "net.4.bias", "gate.2.weight"))
        or not (initial["residual"]["gate.2.bias"] == -4).all()
    ):
        raise ValueError("initial residual output must be exactly zero, not an assisted motor")
    residual.load_state_dict(state["residual"], strict=True)
    scale = torch.from_numpy(
        PaidResearchPolicy(policy, expected_hash=PUBLISHED_HASH, horizon=1)
        .parameters["action_scale"]
        .copy()
    )
    return FrozenContactResidualActor(base, residual, scale).eval(), receipt
