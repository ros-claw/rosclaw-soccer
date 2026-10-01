"""Explicit narrow-kernel experiment around a frozen trained-head transfer.

Chosen after consumed-only attenuation failure. No new optimizer work, no
unseen guarantee. The old module/artifact stays byte-identical and reloadable.
"""

import copy
from pathlib import Path
from typing import Any

from rosclaw.growth.anchor_kernel import AnchorKernelGuard

from rosclaw_soccer.rsi import memory_guarded_phase_transfer as inherited
from rosclaw_soccer.rsi.protected_phase_step_network import FLAGS
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "soccer.rsi.selective_phase_memory.v1"


def validate_model(model: dict[str, Any]) -> None:
    if (
        model.get("schema") != SCHEMA
        or model.get("activation_ceiling") != "SIM_ONLY"
        or model.get("source_hash") != hash_bytes(Path(__file__).read_bytes())
        or model.get("bandwidth") != 1e-4
        or model.get("new_optimizer_steps") != 0
        or model.get("parameter_choice_partition") != "TRAIN_CONSUMED"
        or any(model.get(k) is not False for k in FLAGS)
        or model.get("model_hash")
        != hash_json({k: v for k, v in model.items() if k != "model_hash"})
    ):
        raise ValueError("unsealed explicit selective-memory counterfactual")
    inherited.validate_model(model["transfer_model"])


def make_model(transfer: dict[str, Any]) -> dict[str, Any]:
    inherited.validate_model(transfer)
    value = dict(
        schema=SCHEMA,
        activation_ceiling="SIM_ONLY",
        transfer_model=copy.deepcopy(transfer),
        bandwidth=1e-4,
        new_optimizer_steps=0,
        parameter_choice_partition="TRAIN_CONSUMED",
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        **dict.fromkeys(FLAGS, False),
    )
    value["model_hash"] = hash_json(value)
    validate_model(value)
    return value


def make_preview(model: dict[str, Any]) -> dict[str, Any]:
    validate_model(model)
    policy = inherited.make_preview(model["transfer_model"])
    policy["training_commitment"] = model["model_hash"]
    policy["step_motor_proof"] = {
        **policy["step_motor_proof"],
        "model": model,
        "execution_source_hash": hash_bytes(Path(__file__).read_bytes()),
    }
    policy.pop("memory_phase_motor_proof")
    policy["selective_memory_motor_proof"] = dict(
        transfer_model_hash=model["transfer_model"]["model_hash"],
        bandwidth=model["bandwidth"],
        new_optimizer_steps=0,
        promotion_authorized=False,
    )
    policy.pop("policy_hash")
    policy["policy_hash"] = hash_json(policy)
    return policy


class CompiledSelectivePhaseMemory(inherited.CompiledMemoryPhaseMotor):
    def __init__(self, policy: dict[str, Any]) -> None:
        model = copy.deepcopy(policy["step_motor_proof"]["model"])
        if make_preview(model) != policy:
            raise ValueError("selective-memory execution binding changed")
        transfer = model["transfer_model"]
        super().__init__(inherited.make_preview(transfer))
        # Constructor-only configuration of this private decoder instance.
        # No running policy, inherited artifact or external model is mutated.
        self._guard = AnchorKernelGuard(
            transfer["anchor_guard"]["anchors"], bandwidth=model["bandwidth"]
        )
        self._policy_hash = policy["policy_hash"]
