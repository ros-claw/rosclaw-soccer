"""Load one independently audited whole receiver only for SIM evaluation."""

import json
from pathlib import Path

import torch

from rosclaw_soccer.providers.g1.velocity_policy import file_hash
from rosclaw_soccer.rsi.relay_whole_receiver_identity import zero_receiver_actor


def load_research_receiver(
    setup,
    checkpoint: Path,
    checkpoint_hash: str,
    audit: Path,
    audit_hash: str,
    dataset: Path,
    dataset_hash: str,
):
    for path, expected in (
        (checkpoint, checkpoint_hash),
        (audit, audit_hash),
        (dataset, dataset_hash),
    ):
        if path.is_symlink() or file_hash(path) != expected:
            raise ValueError("exact independently audited candidate inputs required")
    review = json.loads(audit.read_text())
    source = json.loads(dataset.read_text())
    if (
        review.get("status") != "RECEIVER_OFFLINE_CANDIDATE_READBACK_MATCH"
        or review.get("checkpoint_hash") != checkpoint_hash
        or review.get("episodes") != 72
        or review.get("sampled_controls") != 18482
        or review.get("physical_retention_proven") is not False
        or review.get("promotion_authorized") is not False
        or review.get("hardware_authorized") is not False
        or source.get("model_identity")
        != [
            str(setup.velocity_path),
            setup.pins[str(setup.velocity_path)],
            str(setup.velocity_config),
            setup.pins[str(setup.velocity_config)],
        ]
    ):
        raise ValueError("exact audited candidate and velocity reference required")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        state.get("dataset_hash") != dataset_hash
        or state.get("promotion_authorized") is not False
        or state.get("hardware_authorized") is not False
        or state.get("physical_retention_proven") is not False
    ):
        raise ValueError("unpromoted SIM research checkpoint required")
    actor = zero_receiver_actor(setup)
    actor.plastic.load_state_dict(state["actor"], strict=True)
    if any(not torch.isfinite(p).all() for p in actor.parameters()):
        raise ValueError("finite candidate required")
    for path, expected in (
        (checkpoint, checkpoint_hash),
        (audit, audit_hash),
        (dataset, dataset_hash),
    ):
        if file_hash(path) != expected:
            raise ValueError("candidate evidence changed during loading")
    return actor
