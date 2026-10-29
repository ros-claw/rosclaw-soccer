"""SIM-only B continuation checkpoint with explicit parent and codec lineage."""

import json
from pathlib import Path

import torch

from rosclaw_soccer.providers.g1.velocity_policy import file_hash
from rosclaw_soccer.rsi.precontact_receiving_motor import PrecontactReceivingMotor
from rosclaw_soccer.rsi.relay_whole_receiver_identity import zero_receiver_actor


def load_b_child_velocity_candidate(
    setup,
    checkpoint: Path,
    checkpoint_hash: str,
    receipt: Path,
    receipt_hash: str,
    *,
    parent_hash: str,
):
    for path, expected in ((checkpoint, checkpoint_hash), (receipt, receipt_hash)):
        if path.is_symlink() or file_hash(path) != expected:
            raise ValueError("exact pinned child velocity candidate required")
    info = json.loads(receipt.read_text())
    identity = [
        str(setup.velocity_path),
        setup.pins[str(setup.velocity_path)],
        str(setup.velocity_config),
        setup.pins[str(setup.velocity_config)],
    ]
    if (
        info.get("status") != "B_CHILD_VELOCITY_OWN_HISTORY_CANDIDATE"
        or info.get("checkpoint_hash") != checkpoint_hash
        or info.get("parent_hash") != parent_hash
        or info.get("model_identity") != identity
        or info.get("episodes") != 48
        or info.get("actor_result", {}).get("accepted") is not True
        or info.get("actor_result", {}).get("actor_updates") != 1
        or any(
            info.get(k) is not False
            for k in ("physical_retention_proven", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("exact accepted unpromoted own-history child candidate required")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or state.get("parent_hash") != parent_hash
        or state.get("model_identity") != identity
        or state.get("audit_hashes") != info.get("audit_hashes")
        or state.get("hardware_authorized") is not False
        or state.get("promotion_authorized") is not False
    ):
        raise ValueError("exact child lineage and safety ceiling required")
    zero = zero_receiver_actor(setup)
    actor = PrecontactReceivingMotor(zero.plastic, zero.action_scale).eval().requires_grad_(False)
    actor.plastic.load_state_dict(state["actor"], strict=True)
    if any(not torch.isfinite(p).all() for p in actor.parameters()):
        raise ValueError("finite child candidate weights required")
    if any(
        file_hash(path) != expected
        for path, expected in ((checkpoint, checkpoint_hash), (receipt, receipt_hash))
    ):
        raise ValueError("candidate evidence changed during loading")
    return actor
