import json
from types import SimpleNamespace

import pytest
import torch
from test_velocity_prior_actor import tensors

from rosclaw_soccer.providers.g1.velocity_policy import file_hash
from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor
from rosclaw_soccer.rsi.relay_receiver_candidate import load_research_receiver
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor


def test_candidate_loader_requires_exact_audit_and_model_identity(tmp_path, monkeypatch):
    model, config = tmp_path / "model.onnx", tmp_path / "config.yaml"
    setup = SimpleNamespace(
        velocity_path=model, velocity_config=config, pins={str(model): "m", str(config): "c"}
    )
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"model_identity": [str(model), "m", str(config), "c"]}))
    actor = (
        AnchoredReceivingMotor(
            VelocityPriorActor(tensors(), reference_features=14), torch.full((29,), 0.25)
        )
        .eval()
        .requires_grad_(False)
    )
    monkeypatch.setattr(
        "rosclaw_soccer.rsi.relay_receiver_candidate.zero_receiver_actor", lambda _: actor
    )
    checkpoint = tmp_path / "checkpoint.pt"
    state = {k: v.clone() for k, v in actor.plastic.state_dict().items()}
    state["mlp.6.bias"][0] += 0.01
    torch.save(
        dict(
            actor=state,
            dataset_hash=file_hash(dataset),
            promotion_authorized=False,
            hardware_authorized=False,
            physical_retention_proven=False,
        ),
        checkpoint,
    )
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            dict(
                status="RECEIVER_OFFLINE_CANDIDATE_READBACK_MATCH",
                checkpoint_hash=file_hash(checkpoint),
                episodes=72,
                sampled_controls=18482,
                physical_retention_proven=False,
                promotion_authorized=False,
                hardware_authorized=False,
            )
        )
    )
    result = load_research_receiver(
        setup,
        checkpoint,
        file_hash(checkpoint),
        audit,
        file_hash(audit),
        dataset,
        file_hash(dataset),
    )
    assert torch.equal(result.plastic.state_dict()["mlp.6.bias"], state["mlp.6.bias"])
    assert all(not p.requires_grad for p in result.parameters())
    with pytest.raises(ValueError, match="exact independently audited"):
        load_research_receiver(
            setup, checkpoint, "sha256:wrong", audit, file_hash(audit), dataset, file_hash(dataset)
        )
    dataset.write_text(
        json.dumps(
            {
                "model_identity": [str(model), "m", str(config), "c"],
                "reward_profile": "center-strict-full-history-v1",
            }
        )
    )
    torch.save(
        dict(
            actor=state,
            dataset_hash=file_hash(dataset),
            dense_audit_hash="dense-audit",
            promotion_authorized=False,
            hardware_authorized=False,
            physical_retention_proven=False,
        ),
        checkpoint,
    )
    audit.write_text(
        json.dumps(
            dict(
                status="DENSE_RECEIVER_CANDIDATE_READBACK_MATCH",
                checkpoint_hash=file_hash(checkpoint),
                dataset_hash=file_hash(dataset),
                dense_audit_hash="dense-audit",
                episodes=72,
                sampled_controls=18903,
                physical_retention_proven=False,
                promotion_authorized=False,
                hardware_authorized=False,
            )
        )
    )
    args = (
        setup,
        checkpoint,
        file_hash(checkpoint),
        audit,
        file_hash(audit),
        dataset,
        file_hash(dataset),
    )
    with pytest.raises(ValueError, match="exact audited candidate"):
        load_research_receiver(*args)
    assert load_research_receiver(*args, dense_mode=True) is actor
    torch.save(
        dict(
            actor=state,
            dataset_hash=file_hash(dataset),
            dense_audit_hash="motion-audit",
            feedback_profile="receiver-motion-kinematic-v2",
            promotion_authorized=False,
            hardware_authorized=False,
            physical_retention_proven=False,
        ),
        checkpoint,
    )
    audit.write_text(
        json.dumps(
            dict(
                status="MOTION_RECEIVER_CANDIDATE_READBACK_MATCH",
                checkpoint_hash=file_hash(checkpoint),
                dataset_hash=file_hash(dataset),
                dense_audit_hash="motion-audit",
                feedback_profile="receiver-motion-kinematic-v2",
                episodes=72,
                sampled_controls=18903,
                physical_retention_proven=False,
                promotion_authorized=False,
                hardware_authorized=False,
            )
        )
    )
    args = (
        setup,
        checkpoint,
        file_hash(checkpoint),
        audit,
        file_hash(audit),
        dataset,
        file_hash(dataset),
    )
    with pytest.raises(ValueError, match="exact audited candidate"):
        load_research_receiver(*args, dense_mode=True)
    with pytest.raises(ValueError, match="explicit simulation"):
        load_research_receiver(*args, motion_mode=True)
    assert load_research_receiver(*args, dense_mode=True, motion_mode=True) is actor


def test_continuation_requires_exact_unpromoted_lineage(tmp_path, monkeypatch):
    model, config = tmp_path / "model.onnx", tmp_path / "config.yaml"
    setup = SimpleNamespace(
        velocity_path=model, velocity_config=config, pins={str(model): "m", str(config): "c"}
    )
    dataset, checkpoint, audit = (tmp_path / n for n in ("data.json", "candidate.pt", "audit.json"))
    dataset.write_text(
        json.dumps(
            dict(
                status="LEARNED_RECEIVER_PILOT_PHYSICAL_DATA_NOT_TRAINED",
                sampled_controls=123,
                model_identity=[str(model), "m", str(config), "c"],
            )
        )
    )
    actor = (
        AnchoredReceivingMotor(
            VelocityPriorActor(tensors(), reference_features=14), torch.full((29,), 0.25)
        )
        .eval()
        .requires_grad_(False)
    )
    monkeypatch.setattr(
        "rosclaw_soccer.rsi.relay_receiver_candidate.zero_receiver_actor", lambda _: actor
    )
    state = actor.plastic.state_dict()
    torch.save(
        dict(
            actor=state,
            dataset_hash=file_hash(dataset),
            dataset_audit_hash="data-audit",
            parent_hash="parent",
            physical_retention_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
        checkpoint,
    )
    review = dict(
        status="LEARNED_RECEIVER_CONTINUATION_NUMERIC_READBACK_MATCH",
        checkpoint_hash=file_hash(checkpoint),
        dataset_hash=file_hash(dataset),
        dataset_audit_hash="data-audit",
        parent_hash="parent",
        episodes=24,
        sampled_controls=123,
        physical_retention_proven=False,
        independent_new_physics=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    audit.write_text(json.dumps(review))
    args = (
        setup,
        checkpoint,
        file_hash(checkpoint),
        audit,
        file_hash(audit),
        dataset,
        file_hash(dataset),
    )
    assert load_research_receiver(*args, dense_mode=True, continuation_mode=True) is actor
    with pytest.raises(ValueError, match="explicit simulation"):
        load_research_receiver(*args, continuation_mode=True)
    review["parent_hash"] = "wrong"
    audit.write_text(json.dumps(review))
    args = (
        setup,
        checkpoint,
        file_hash(checkpoint),
        audit,
        file_hash(audit),
        dataset,
        file_hash(dataset),
    )
    with pytest.raises(ValueError, match="continuation checkpoint"):
        load_research_receiver(*args, dense_mode=True, continuation_mode=True)
