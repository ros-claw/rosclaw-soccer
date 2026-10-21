"""Separate fail-closed CPU reader for complete-world sequence research weights.

No old checkpoint schema is relaxed or recursively admitted here. Exactly one
new sequence layer may warm-start the already qualified live-v2 actor.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from rosclaw_soccer.rsi.paid_live_distilled_actor import load_live_distilled_actor
from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH, PaidTrainableActor


def sequence_config(variant: int) -> dict[str, Any]:
    if type(variant) is not int or variant not in range(4):
        raise ValueError("one predeclared sequence training seed or ablation required")
    return dict(
        protocol="complete-successor-sequences-v1",
        variant=variant,
        seed=20261028 + (variant if variant < 3 else 0),
        epochs=256,
        lr=0.0001,
        first_weight=3.0,
        active_weight=5.0,
        other_weight=1.0,
        max_grad_norm=1.0,
        worlds=16,
        first_unique_rows=3810,
        second_unique_rows=3810,
        minibatches_per_epoch=4,
        worlds_per_minibatch=4,
        memory_handoff="carry" if variant < 3 else "reset",
        full_world_BPTT=True,
        teacher_hidden_states_not_inputs=True,
        normalizer_frozen=True,
        DEV_episodes_used=0,
        observation_gap="no fabricated neural rows; memory constant during hold/drop",
        loss_units="joint_target_radians_squared",
    )


def validate_successor_state(
    state: dict[str, Any],
    receipt: dict[str, Any],
    initial: dict[str, torch.Tensor],
    *,
    declaration_hash: str,
    scene_hash: str,
) -> None:
    config = sequence_config(state.get("config", {}).get("variant"))
    counts = dict(first_forward_rows=975360, second_forward_rows=975360, optimizer_updates=1024)
    if (
        state.get("schema") != "soccer.paid_successor_sequences.v1"
        or state.get("config") != config
        or receipt.get("config") != config
        or state.get("epoch") != 256
        or state.get("counts") != counts
        or receipt.get("counts") != counts
        or state.get("source_actor_hash") != PUBLISHED_HASH
        or state.get("declaration_hash") != declaration_hash
        or state.get("body_scene_hash") != scene_hash
        or state.get("source_head") != receipt.get("source_head")
        or receipt.get("status") != "PAID_SUCCESSOR_SEQUENCE_TRAINED_NOT_PHYSICALLY_EXAMINED"
        or receipt.get("native_steps") != 0
        or receipt.get("CPU_whole_world_exam_required") is not True
        or any(
            item.get(k) is not False
            for item in (state, receipt)
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "commercial_publicity_authorized",
            )
        )
    ):
        raise ValueError(
            "exact completed sequence research protocol and authority ceiling required"
        )
    proofs = set(receipt.get("input_file_hashes", {}).values())
    if any(
        state.get(k) not in proofs
        for k in ("base_checkpoint_hash", "base_training_receipt_hash", "bank_hash")
    ):
        raise ValueError("complete-world data and base commitments must bind real training inputs")
    network = state.get("network", {})
    if not isinstance(network, dict) or set(network) != set(initial):
        raise ValueError("exact full recurrent network keys required")
    for key, expected in initial.items():
        value = network[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or value.layout != torch.strided
            or value.shape != expected.shape
            or not torch.isfinite(value).all()
        ):
            raise ValueError("finite exact CPU actor tensor required: " + key)
    if any(not torch.equal(network[k], initial[k]) for k in ("mean", "denominator")):
        raise ValueError("original observation normalization must not change")
    optimizer = state.get("optimizer", {})
    groups = optimizer.get("param_groups", [])
    names = [k for k in initial if k not in ("mean", "denominator")]
    if (
        len(names) != 16
        or len(groups) != 1
        or groups[0].get("params") != list(range(16))
        or groups[0].get("lr") != 0.0001
        or set(optimizer.get("state", {})) != set(range(16))
    ):
        raise ValueError("all sixteen complete actor Adam parameter slots required")
    for pid, key in enumerate(names):
        slot = optimizer["state"][pid]
        step = slot.get("step")
        if (
            not isinstance(step, torch.Tensor)
            or step.device.type != "cpu"
            or step.numel() != 1
            or float(step) != 1024
        ):
            raise ValueError("actual completed 1024 Adam updates required")
        for field in ("exp_avg", "exp_avg_sq"):
            value = slot.get(field)
            if (
                not isinstance(value, torch.Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.float32
                or value.layout != torch.strided
                or value.shape != initial[key].shape
                or not torch.isfinite(value).all()
                or (field == "exp_avg_sq" and (value < 0).any())
            ):
                raise ValueError("finite shape-bound safe Adam slot required")


def load_successor_actor(
    policy: Path,
    *,
    checkpoint: Path,
    checkpoint_hash: str,
    training_receipt: Path,
    training_receipt_hash: str,
    scene_hash: str,
) -> tuple[PaidTrainableActor, dict[str, Any]]:
    def digest(path: Path) -> str:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("bounded ordinary source checkpoint/receipt required")
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    if digest(checkpoint) != checkpoint_hash or digest(training_receipt) != training_receipt_hash:
        raise ValueError("exact completed checkpoint/receipt required")
    receipt = json.loads(training_receipt.read_text())
    if receipt.get("final_checkpoint_hash") != checkpoint_hash:
        raise ValueError("only declared final checkpoint may enter CPU exams")
    dp = training_receipt.parent / "declaration.json"
    dh = digest(dp)
    declaration = json.loads(dp.read_text())
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("safe typed sequence checkpoint required")
    if (
        declaration.get("input_file_hashes") != receipt.get("input_file_hashes")
        or declaration.get("config") != receipt.get("config")
        or [w["case"] for w in declaration.get("worlds", [])] != list(range(16))
        or declaration.get("original160_observation_contract_unchanged") is not True
        or declaration.get("no_case_id_policy_inputs") is not True
        or declaration.get("no_runtime_teacher") is not True
    ):
        raise ValueError("exact complete-world declaration required")
    pins = receipt["input_file_hashes"]

    def unique(key: str) -> Path:
        paths = [Path(p) for p, h in pins.items() if h == state.get(key)]
        if len(paths) != 1 or digest(paths[0]) != state[key]:
            raise ValueError("one exact source-bound base artifact required")
        return paths[0]

    base_cp, base_rp, bank_path = (
        unique("base_checkpoint_hash"),
        unique("base_training_receipt_hash"),
        unique("bank_hash"),
    )
    base_state = torch.load(base_cp, map_location="cpu", weights_only=True)
    if base_state.get("schema") != "soccer.paid_live_distilled.v2":
        raise ValueError(
            "exact one-layer live-v2 warmstart required; no recursive sequence admission"
        )
    base, _ = load_live_distilled_actor(
        policy,
        checkpoint=base_cp,
        checkpoint_hash=state["base_checkpoint_hash"],
        training_receipt=base_rp,
        training_receipt_hash=state["base_training_receipt_hash"],
        scene_hash=scene_hash,
    )
    bank = json.loads(bank_path.read_text())
    if (
        bank.get("schema") != "soccer.audited_successor_sequence_bank.v1"
        or bank.get("courses") != declaration["worlds"]
        or bank.get("base_checkpoint_hash") != state["base_checkpoint_hash"]
        or bank.get("complete_successful_course_coverage") != 16
        or bank.get("body_reset_between_skills") is not False
        or bank.get("single_policy_16of16_proven") is not False
        or bank.get("runtime_case_lookup_forbidden") is not True
    ):
        raise ValueError("same complete-world offline data and qualified initialization required")
    validate_successor_state(
        state, receipt, base.state_dict(), declaration_hash=dh, scene_hash=scene_hash
    )
    start_path = training_receipt.parent / "successor-epoch0.pt"
    if digest(start_path) != receipt.get("initial_checkpoint_hash"):
        raise ValueError("actual initialization checkpoint identity differs")
    start = torch.load(start_path, map_location="cpu", weights_only=True)
    if (
        start.get("epoch") != 0
        or start.get("schema") != state["schema"]
        or start.get("declaration_hash") != dh
        or start.get("base_checkpoint_hash") != state["base_checkpoint_hash"]
        or set(start.get("network", {})) != set(base.state_dict())
        or any(not torch.equal(start["network"][k], v) for k, v in base.state_dict().items())
    ):
        raise ValueError("exact physically qualified complete live-v2 network warmstart required")
    base.load_state_dict(state["network"], strict=True)
    return base.eval(), receipt
