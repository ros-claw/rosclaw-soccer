"""Fail-closed CPU admission of the fixed16 Cartesian-distillation protocol.

Loading grants no skill, physics, promotion or hardware authority. The model
outputs joint targets through the separately qualified original PD contract;
it does NOT call IK, read a latent intent or output raw motor torque.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH, PaidTrainableActor


def validate_distilled_state(
    state: dict[str, Any],
    receipt: dict[str, Any],
    initial: dict[str, torch.Tensor],
    *,
    declaration_hash: str,
    scene_hash: str,
) -> None:
    config = state.get("config", {})
    epoch = state.get("epoch")
    if (
        state.get("schema") != "soccer.paid_cartesian_distilled.v1"
        or type(epoch) is not int
        or epoch not in (128, 256)
        or receipt.get("status") != "PAID_CARTESIAN_DISTILLATION_TRAINED_NOT_PHYSICALLY_EXAMINED"
        or receipt.get("config") != config
        or config.get("epochs") != epoch
        or config.get("teacher_episodes") != 16
        or config.get("DEV_episodes_used") != 0
        or state.get("source_actor_hash") != PUBLISHED_HASH
        or state.get("declaration_hash") != declaration_hash
        or state.get("body_scene_hash") != scene_hash
        or state.get("source_head") != receipt.get("source_head")
        or receipt.get("native_steps") != 0
        or receipt.get("CPU_neural_only_exam_required") is not True
    ):
        raise ValueError(
            "exact completed fixed16 research distillation/body/source contract required"
        )
    expected_counts = dict(
        optimizer_updates=4 * epoch,
        teacher_forward_rows=3810 * epoch,
        rehearsal_forward_rows=3810 * epoch,
    )
    if state.get("counts") != expected_counts or receipt.get("counts") != expected_counts:
        raise ValueError("actual full16 fixed-source training/update counts required")
    proofs = set(receipt.get("input_file_hashes", {}).values())
    if state.get("teacher_hash") not in proofs or state.get("teacher_audit_hash") not in proofs:
        raise ValueError("candidate teacher/data audit commitments must bind the training inputs")
    if any(
        state.get(k) is not False or receipt.get(k) is not False
        for k in ("hardware_authorized", "promotion_authorized", "commercial_publicity_authorized")
    ):
        raise ValueError("research-only SIM/noncommercial ceiling required")
    network = state.get("network")
    if not isinstance(network, dict) or set(network) != set(initial):
        raise ValueError("exact full recurrent actor network keys required")
    for key, expected in initial.items():
        value = network[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or value.shape != expected.shape
            or not torch.isfinite(value).all()
        ):
            raise ValueError("finite exact CPU float32 actor tensor required: " + key)
    if any(not torch.equal(network[k], initial[k]) for k in ("mean", "denominator")):
        raise ValueError("original immutable normalization required")
    optimizer = state.get("optimizer", {})
    groups = optimizer.get("param_groups", [])
    expected_slots = 8 if config.get("freeze_rnn") is True else 16
    if len(groups) != 1 or len(groups[0].get("params", [])) != expected_slots:
        raise ValueError("exact declared trainable Adam parameter slots required")
    for pid in groups[0]["params"]:
        moment = optimizer.get("state", {}).get(pid)
        if (
            moment is None
            or float(moment["step"]) != 4 * epoch
            or not all(torch.isfinite(moment[k]).all() for k in ("exp_avg", "exp_avg_sq"))
        ):
            raise ValueError("actual safe Adam update state required")


def load_distilled_actor(
    policy: Path,
    *,
    checkpoint: Path,
    checkpoint_hash: str,
    training_receipt: Path,
    training_receipt_hash: str,
    scene_hash: str,
) -> tuple[PaidTrainableActor, dict[str, Any]]:
    def digest(path: Path, *, maximum_bytes: int) -> str:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > maximum_bytes:
            raise ValueError("bounded ordinary completed model/evidence file required")
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    if (
        digest(checkpoint, maximum_bytes=64 * 1024 * 1024) != checkpoint_hash
        or digest(training_receipt, maximum_bytes=32 * 1024 * 1024) != training_receipt_hash
    ):
        raise ValueError("exact candidate/training receipt commitments required")
    receipt = json.loads(training_receipt.read_text())
    if receipt.get("final_checkpoint_hash") != checkpoint_hash:
        raise ValueError("only the predeclared final checkpoint may enter this exam")
    declaration = training_receipt.parent / "declaration.json"
    dh = digest(declaration, maximum_bytes=32 * 1024 * 1024)
    contract = json.loads(declaration.read_text())
    if contract.get("config") != receipt.get("config") or len(contract.get("data", [])) != 16:
        raise ValueError("immutable complete training-only cohort declaration required")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("safe typed checkpoint mapping required")
    actor = PaidTrainableActor(policy, expected_hash=PUBLISHED_HASH)
    validate_distilled_state(
        state, receipt, actor.state_dict(), declaration_hash=dh, scene_hash=scene_hash
    )
    actor.load_state_dict(state["network"], strict=True)
    actor.eval()
    return actor, receipt
