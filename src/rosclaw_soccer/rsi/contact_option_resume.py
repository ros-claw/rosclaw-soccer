"""Exact boundary state restoration for the fixed native finite-option pilot.

Restores model/Adam/RNG only. No physics, model construction, reset, inference,
gradient update or policy promotion. Caller owns the subsequent finite budget.
"""

import hashlib
from pathlib import Path
from typing import Any

import torch

IDENTITY_KEYS = frozenset(
    (
        "parent_checkpoint_hash",
        "original_onnx_hash",
        "parameter_hash",
        "native_model_hash",
        "reset_bank_hash",
        "adapter_proof_hash",
        "recovery_proof_hash",
        "recovery_fault_hash",
    )
)


def restore_contact_option_boundary(
    policy: Any,
    optimizer: Any,
    path: Path,
    *,
    expected_hash: str,
    expected_identity: dict[str, str],
) -> dict[str, Any]:
    """Fresh same-architecture objects, qualified original identity, checkpoint20."""

    def digest() -> str:
        return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    if set(expected_identity) != IDENTITY_KEYS or digest() != expected_hash or optimizer.state:
        raise ValueError("exact qualified identities, checkpoint, and unused optimizer required")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (
        state.get("schema") != "soccer.rsi.native_contact24_recovery60_residual_ppo.v1"
        or state.get("protocol") != "native-contact24-recovery60-v1"
        or state.get("training_seed") != 7607
        or state.get("episode_count") != 20
        or state.get("course_episode") != 19
        or state.get("fresh_optimizer") is not True
        or state.get("activation_ceiling") != "SIM_ONLY"
        or state.get("hardware_authorized") is not False
        or state.get("promotion_authorized") is not False
        or any(state.get(k) != v for k, v in expected_identity.items())
    ):
        raise ValueError("fixed complete finite-option pilot boundary required")
    counts = state["counts"]
    if (
        counts["optimizer_attempted"] != 320
        or counts["optimizer_completed"] != 320
        or counts["value_only_steps"] != 64
        or counts["ppo_steps"] != 256
    ):
        raise ValueError("actual complete critic/PPO pilot optimizer history required")
    weights = state["policy_state_dict"]
    current = policy.state_dict()
    if set(weights) != set(current) or any(
        not torch.isfinite(v).all() or v.shape != current[k].shape or v.dtype != current[k].dtype
        for k, v in weights.items()
    ):
        raise ValueError("complete finite same-architecture model state required")
    for k, v in current.items():
        if (k.startswith("baseline.") or k.startswith("actor.base.")) and not torch.equal(
            v.cpu(), weights[k]
        ):
            raise ValueError("checkpoint changed immutable original foundation/baseline")
    groups = state["optimizer_state_dict"]["param_groups"]
    if (
        len(groups) != 3
        or [x["lr"] for x in groups] != [3e-4, 3e-4, 1e-5]
        or [len(x["params"]) for x in groups] != [6, 6, 1]
    ):
        raise ValueError("exact adapter/critic/std Adam layout required")
    slots = state["optimizer_state_dict"]["state"]
    ids = [i for group in groups for i in group["params"]]
    if len(set(ids)) != 13 or set(slots) != set(ids):
        raise ValueError("complete nonduplicated Adam moments required")
    for group_index, group in enumerate(groups):
        expected_step = 320 if group_index == 1 else 256
        for i in group["params"]:
            if (
                set(slots[i]) != {"step", "exp_avg", "exp_avg_sq"}
                or float(slots[i]["step"]) != expected_step
            ):
                raise ValueError("Adam per-parameter steps do not match actual pilot updates")
    for values in slots.values():
        if any(
            isinstance(v, torch.Tensor) and not torch.isfinite(v).all() for v in values.values()
        ):
            raise ValueError("finite Adam state required")
    cpu_rng, cuda_rng = state["torch_rng"], state["cuda_rng"]
    if (
        cpu_rng.dtype != torch.uint8
        or cpu_rng.ndim != 1
        or len(cuda_rng) != 1
        or torch.cuda.device_count() != 1
        or any(x.dtype != torch.uint8 or x.ndim != 1 for x in cuda_rng)
    ):
        raise ValueError("exact CPU and one isolated CUDA RNG stream required")
    if digest() != expected_hash:
        raise ValueError("checkpoint changed during safe read")
    policy.load_state_dict(weights, strict=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    # Do this last: construction/admission must not consume the saved stream.
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    return dict(
        counts=dict(counts),
        course_episode=19,
        episode_count=20,
        source_checkpoint_hash=expected_hash,
        training_seed=7607,
        optimizer_resumed=True,
        goal_complete=False,
        hardware_authorized=False,
    )
