"""Fail-closed CPU admission for the separate live-body BC/rehearsal protocol.

The original fixed16 loader remains unchanged. Loading these research weights
does not prove a skill, promote a model, or grant physical-robot authority.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from rosclaw_soccer.rsi.paid_distilled_actor import load_distilled_actor
from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH, PaidTrainableActor


def validate_live_state(
    state: dict[str, Any],
    receipt: dict[str, Any],
    initial: dict[str, torch.Tensor],
    *,
    declaration_hash: str,
    scene_hash: str,
) -> None:
    config = state.get("config", {})
    variant = config.get("variant")
    counts = dict(
        optimizer_updates=768, teacher_forward_rows=2898 * 256, rehearsal_forward_rows=3810 * 256
    )
    if (
        state.get("schema") != "soccer.paid_live_distilled.v1"
        or state.get("epoch") != 256
        or receipt.get("status") != "PAID_LIVE_DISTILLATION_TRAINED_NOT_PHYSICALLY_EXAMINED"
        or receipt.get("config") != config
        or config.get("epochs") != 256
        or type(variant) is not int
        or variant not in range(4)
        or config.get("seed") != 20261022 + (variant if variant < 3 else 0)
        or config.get("lr") != 0.0001
        or config.get("rehearsal") != (1.0 if variant < 3 else 3.0)
        or config.get("freeze_rnn") is not False
        or config.get("teacher_episodes") != 12
        or config.get("rehearsal_episodes") != 16
        or config.get("exam_continuous_cases") != 16
        or config.get("DEV_episodes_used") != 0
        or config.get("unique_teacher_neural_rows") != 2898
        or config.get("unique_rehearsal_neural_rows") != 3810
        or config.get("full_episode_BPTT") is not True
        or config.get("teacher_hidden_states_not_inputs") is not True
        or config.get("normalizer_frozen") is not True
        or config.get("active_weight") != 5.0
        or config.get("other_weight") != 1.0
        or config.get("max_grad_norm") != 1.0
        or config.get("loss_units") != "joint_target_radians_squared"
        or config.get("teacher_minibatches_per_epoch") != 3
        or config.get("rehearsal_minibatch_sizes") != [6, 5, 5]
        or state.get("source_actor_hash") != PUBLISHED_HASH
        or state.get("declaration_hash") != declaration_hash
        or state.get("body_scene_hash") != scene_hash
        or state.get("source_head") != receipt.get("source_head")
        or state.get("counts") != counts
        or receipt.get("counts") != counts
        or receipt.get("native_steps") != 0
        or receipt.get("CPU_neural_only_full_sequence_exam_required") is not True
    ):
        raise ValueError("exact completed live12/rehearsal16 research protocol required")
    proofs = set(receipt.get("input_file_hashes", {}).values())
    if any(
        state.get(key) not in proofs
        for key in (
            "base_checkpoint_hash",
            "base_training_receipt_hash",
            "teacher_hash",
            "teacher_audit_hash",
        )
    ):
        raise ValueError("base model and audited live teacher must bind actual training inputs")
    if any(
        state.get(k) is not False or receipt.get(k) is not False
        for k in ("hardware_authorized", "promotion_authorized", "commercial_publicity_authorized")
    ):
        raise ValueError("SIM research-only authority ceiling required")
    network = state.get("network", {})
    if not isinstance(network, dict) or set(network) != set(initial):
        raise ValueError("exact complete recurrent network keys required")
    for k, expected in initial.items():
        x = network[k]
        if (
            not isinstance(x, torch.Tensor)
            or x.device.type != "cpu"
            or x.dtype != torch.float32
            or x.layout != torch.strided
            or x.shape != expected.shape
            or not torch.isfinite(x).all()
        ):
            raise ValueError("finite exact CPU actor tensor required: " + k)
    if any(not torch.equal(network[k], initial[k]) for k in ("mean", "denominator")):
        raise ValueError("immutable original normalization required")
    optimizer = state.get("optimizer", {})
    groups = optimizer.get("param_groups", [])
    names = [k for k in initial if k not in ("mean", "denominator")]
    if (
        len(names) != 16
        or len(groups) != 1
        or groups[0].get("params") != list(range(16))
        or groups[0].get("lr") != config["lr"]
        or len(optimizer.get("state", {})) != 16
    ):
        raise ValueError("all sixteen full actor Adam slots required")
    for pid, k in enumerate(names):
        moment = optimizer["state"].get(pid, {})
        step = moment.get("step")
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or float(step) != 768:
            raise ValueError("actual completed Adam step count required")
        for field in ("exp_avg", "exp_avg_sq"):
            x = moment.get(field)
            if (
                not isinstance(x, torch.Tensor)
                or x.device.type != "cpu"
                or x.dtype != torch.float32
                or x.shape != initial[k].shape
                or not torch.isfinite(x).all()
                or (field == "exp_avg_sq" and (x < 0).any())
            ):
                raise ValueError("finite shape-bound safe Adam state required")


def load_live_distilled_actor(
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
            raise ValueError("bounded ordinary checkpoint/evidence required")
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    if digest(checkpoint) != checkpoint_hash or digest(training_receipt) != training_receipt_hash:
        raise ValueError("exact completed checkpoint/training commitments required")
    receipt = json.loads(training_receipt.read_text())
    if receipt.get("final_checkpoint_hash") != checkpoint_hash:
        raise ValueError("only the predeclared final checkpoint may enter this exam")
    dp = training_receipt.parent / "declaration.json"
    dh = digest(dp)
    declaration = json.loads(dp.read_text())
    if (
        declaration.get("config") != receipt.get("config")
        or declaration.get("input_file_hashes") != receipt.get("input_file_hashes")
        or [x["case"] for x in declaration.get("expert", [])]
        != [0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 13, 14]
        or [x["case"] for x in declaration.get("rehearsal", [])] != list(range(16))
        or declaration.get("excluded_positive_label_cases") != [7, 10, 11, 15]
        or len(declaration.get("all16_teacher_outcomes", [])) != 16
        or declaration.get("all16_continuous_cases_remain_in_exam") is not True
    ):
        raise ValueError(
            "complete unfiltered curriculum plus explicit positive-label mask required"
        )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("safe typed live-model checkpoint required")
    actor = PaidTrainableActor(policy, expected_hash=PUBLISHED_HASH)
    validate_live_state(
        state, receipt, actor.state_dict(), declaration_hash=dh, scene_hash=scene_hash
    )
    pins = receipt["input_file_hashes"]

    def unique_path(hash_key: str) -> Path:
        paths = [Path(path) for path, h in pins.items() if h == state[hash_key]]
        if len(paths) != 1:
            raise ValueError("one exact source-bound base artifact required")
        return paths[0]

    base, _ = load_distilled_actor(
        policy,
        checkpoint=unique_path("base_checkpoint_hash"),
        checkpoint_hash=state["base_checkpoint_hash"],
        training_receipt=unique_path("base_training_receipt_hash"),
        training_receipt_hash=state["base_training_receipt_hash"],
        scene_hash=scene_hash,
    )
    initial_path = training_receipt.parent / "live-distilled-epoch0.pt"
    if digest(initial_path) != receipt.get("initial_checkpoint_hash"):
        raise ValueError("actual warm-start checkpoint commitment differs")
    start = torch.load(initial_path, map_location="cpu", weights_only=True)
    if (
        start.get("epoch") != 0
        or start.get("declaration_hash") != dh
        or start.get("base_checkpoint_hash") != state["base_checkpoint_hash"]
        or set(start.get("network", {})) != set(base.state_dict())
        or any(not torch.equal(start["network"][k], v) for k, v in base.state_dict().items())
    ):
        raise ValueError("training must warm-start the physically qualified full v0 network")
    actor.load_state_dict(state["network"], strict=True)
    actor.eval()
    return actor, receipt
