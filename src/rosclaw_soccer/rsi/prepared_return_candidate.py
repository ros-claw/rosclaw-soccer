"""Load only a pinned, unpromoted SIM return-player research checkpoint."""

import json
from pathlib import Path

import torch

from rosclaw_soccer.providers.g1.velocity_policy import file_hash
from rosclaw_soccer.rsi.precontact_receiving_motor import PrecontactReceivingMotor
from rosclaw_soccer.rsi.relay_whole_receiver_identity import zero_receiver_actor


def load_prepared_return_candidate(
    setup, checkpoint: Path, checkpoint_hash: str, receipt: Path, receipt_hash: str
):
    for path, expected in ((checkpoint, checkpoint_hash), (receipt, receipt_hash)):
        if path.is_symlink() or file_hash(path) != expected:
            raise ValueError("exact pinned return research candidate required")
    info = json.loads(receipt.read_text())
    identity = [
        str(setup.velocity_path),
        setup.pins[str(setup.velocity_path)],
        str(setup.velocity_config),
        setup.pins[str(setup.velocity_config)],
    ]
    if (
        info.get("status") != "PREPARED_RETURN_ACTOR_RESEARCH_CANDIDATE"
        or info.get("checkpoint_hash") != checkpoint_hash
        or info.get("model_identity") != identity
        or info.get("actor_result", {}).get("accepted") is not True
        or info.get("actor_result", {}).get("actor_updates") != 1
        or any(
            info.get(k) is not False
            for k in ("physical_retention_proven", "promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("accepted unpromoted exact-motor SIM research candidate required")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or state.get("model_hash") != identity[1]
        or state.get("config_hash") != identity[3]
        or state.get("data_hashes") != info.get("data_hashes")
        or state.get("hardware_authorized") is not False
        or state.get("promotion_authorized") is not False
    ):
        raise ValueError("exact candidate lineage and safety ceiling required")
    zero = zero_receiver_actor(setup)
    actor = PrecontactReceivingMotor(zero.plastic, zero.action_scale).eval().requires_grad_(False)
    actor.plastic.load_state_dict(state["actor"], strict=True)
    if any(not torch.isfinite(p).all() for p in actor.parameters()):
        raise ValueError("finite candidate parameters required")
    if any(
        file_hash(path) != expected
        for path, expected in ((checkpoint, checkpoint_hash), (receipt, receipt_hash))
    ):
        raise ValueError("candidate evidence changed during loading")
    return actor
