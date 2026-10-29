import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_velocity_prior_actor import tensors

from rosclaw_soccer.providers.g1.velocity_policy import file_hash
from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor
from rosclaw_soccer.rsi.prepared_return_candidate import load_prepared_return_candidate
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor


def test_candidate_is_pinned_and_unpromoted(tmp_path, monkeypatch):
    model, config = Path("/fake/model"), Path("/fake/config")
    setup = SimpleNamespace(
        velocity_path=model,
        velocity_config=config,
        pins={str(model): "model-hash", str(config): "config-hash"},
    )
    zero = (
        AnchoredReceivingMotor(VelocityPriorActor(tensors(), reference_features=14), torch.ones(29))
        .eval()
        .requires_grad_(False)
    )
    monkeypatch.setattr(
        "rosclaw_soccer.rsi.prepared_return_candidate.zero_receiver_actor", lambda _: zero
    )
    checkpoint, receipt = tmp_path / "candidate.pt", tmp_path / "receipt.json"
    torch.save(
        dict(
            actor=zero.plastic.state_dict(),
            model_hash="model-hash",
            config_hash="config-hash",
            data_hashes=["data"],
            hardware_authorized=False,
            promotion_authorized=False,
        ),
        checkpoint,
    )
    receipt.write_text(
        json.dumps(
            dict(
                status="PREPARED_RETURN_ACTOR_RESEARCH_CANDIDATE",
                checkpoint_hash=file_hash(checkpoint),
                model_identity=[str(model), "model-hash", str(config), "config-hash"],
                actor_result=dict(accepted=True, actor_updates=1),
                data_hashes=["data"],
                physical_retention_proven=False,
                hardware_authorized=False,
                promotion_authorized=False,
            )
        )
    )
    actor = load_prepared_return_candidate(
        setup, checkpoint, file_hash(checkpoint), receipt, file_hash(receipt)
    )
    assert not actor.training and all(not p.requires_grad for p in actor.parameters())
    with pytest.raises(ValueError, match="pinned"):
        load_prepared_return_candidate(setup, checkpoint, "sha256:bad", receipt, file_hash(receipt))
