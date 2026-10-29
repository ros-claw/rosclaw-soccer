import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_velocity_prior_actor import tensors

from rosclaw_soccer.providers.g1.velocity_policy import file_hash
from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor
from rosclaw_soccer.rsi.b_child_velocity_candidate import load_b_child_velocity_candidate
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor


def test_b_candidate_checks_parent_and_hashes(tmp_path, monkeypatch):
    model, config = Path("/fake/model"), Path("/fake/config")
    identity = [str(model), "mh", str(config), "ch"]
    setup = SimpleNamespace(
        velocity_path=model, velocity_config=config, pins={str(model): "mh", str(config): "ch"}
    )
    zero = (
        AnchoredReceivingMotor(VelocityPriorActor(tensors(), reference_features=14), torch.ones(29))
        .eval()
        .requires_grad_(False)
    )
    monkeypatch.setattr(
        "rosclaw_soccer.rsi.b_child_velocity_candidate.zero_receiver_actor", lambda _: zero
    )
    checkpoint, receipt = tmp_path / "candidate.pt", tmp_path / "receipt.json"
    torch.save(
        dict(
            actor=zero.plastic.state_dict(),
            model_identity=identity,
            parent_hash="parent",
            audit_hashes=["audit"],
            hardware_authorized=False,
            promotion_authorized=False,
        ),
        checkpoint,
    )
    receipt.write_text(
        json.dumps(
            dict(
                status="B_CHILD_VELOCITY_OWN_HISTORY_CANDIDATE",
                checkpoint_hash=file_hash(checkpoint),
                parent_hash="parent",
                model_identity=identity,
                audit_hashes=["audit"],
                episodes=48,
                actor_result=dict(accepted=True, actor_updates=1),
                physical_retention_proven=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
        )
    )
    actor = load_b_child_velocity_candidate(
        setup, checkpoint, file_hash(checkpoint), receipt, file_hash(receipt), parent_hash="parent"
    )
    assert not actor.training and all(not p.requires_grad for p in actor.parameters())
    with pytest.raises(ValueError, match="candidate"):
        load_b_child_velocity_candidate(
            setup,
            checkpoint,
            file_hash(checkpoint),
            receipt,
            file_hash(receipt),
            parent_hash="wrong",
        )
