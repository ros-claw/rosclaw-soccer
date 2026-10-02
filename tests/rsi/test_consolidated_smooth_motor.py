"""Small contract fixtures only; not G1 success or physical replay evidence."""

import copy

import numpy as np
import pytest
from rosclaw.growth.anchor_consolidation import consolidate_unique
from rosclaw.growth.anchor_output_memory import AnchorOutputMemory

from rosclaw_soccer.rsi import consolidated_smooth_motor as motor
from rosclaw_soccer.sim.contracts import hash_json


@pytest.fixture
def model(monkeypatch):
    identity = "sha256:" + "a" * 64
    parent = "sha256:" + "b" * 64
    old = AnchorOutputMemory(
        [np.zeros(135)],
        [np.arange(12.0)],
        bandwidth=1e-4,
        encoder_hash=identity,
        parent_policy_hash="sha256:" + "c" * 64,
        evidence_hash="sha256:" + "d" * 64,
    )
    evidence = dict(
        schema="soccer.rsi.current_parent_success_consolidation_evidence.v1",
        parent_model_hash=parent,
        encoder_hash=identity,
        inherited_memory_hash=old.to_dict()["memory_hash"],
        bank_summary_hash="fixture-bank",
        bank_review_hash="fixture-review",
        selection="ALL_QUALIFIED_CURRENT_PARENT_SUCCESSES_IN_SEALED_ORDER",
        records=[dict(seed=42, lane=0, frames=270)],
        source_hash="fixture",
        consolidation_source_hash="fixture",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    expanded, mapping = consolidate_unique(
        old,
        np.zeros((270, 135)),
        np.tile(np.arange(12.0), (270, 1)),
        parent_policy_hash=parent,
        evidence_hash=hash_json(evidence),
    )
    memory = expanded.to_dict()
    manifest = dict(
        **evidence,
        consolidation_mapping=mapping,
        memory_hash=memory["memory_hash"],
        inherited_frames=1,
        recorded_frames=1,
        added_frames=270,
        independent_contexts=1,
        existing_success_reports_reconstructed=1,
        actual_physical_executions_added=0,
        exact_current_parent_output_reload=True,
        physical_policy_execution_qualified=False,
    )
    manifest["report_hash"] = hash_json(manifest)
    base = dict(
        frozen_parent=dict(
            model_hash=parent,
            frozen_parent={},
            output_memory=dict(memory_hash=old.to_dict()["memory_hash"]),
        ),
        parent_model_hash=parent,
        zero=False,
    )
    monkeypatch.setattr(motor, "validate_base", lambda _: None)
    monkeypatch.setattr(motor, "encoder_identity", lambda _: identity)
    monkeypatch.setattr(motor, "base_preview", lambda base: dict(zero=base["zero"]))

    def initialize(self, policy):
        self._zero = policy["zero"]

    monkeypatch.setattr(motor.CompiledSmoothMemoryMotor, "__init__", initialize)
    monkeypatch.setattr(motor.CompiledSmoothMemoryMotor, "features", lambda self, x: x)
    monkeypatch.setattr(motor.CompiledSmoothMemoryMotor, "raw_mean", lambda *a: np.ones(12) * 99.0)
    return motor.make_model(base, memory, manifest)


def reseal(model):
    model["model_hash"] = hash_json({k: v for k, v in model.items() if k != "model_hash"})


def test_known_parent_output_restored_without_disabling_novel_proposal(model):
    decoder = motor.CompiledConsolidatedSmoothMotor(motor.make_preview(model))
    assert np.array_equal(decoder.raw_mean(np.zeros(134), 0), np.arange(12.0))
    assert np.array_equal(decoder.raw_mean(np.ones(134) * 100, 0), np.ones(12) * 99)


def test_zero_addition_is_globally_exact_and_not_interpolated(model):
    model["base_model"]["zero"] = True
    reseal(model)
    decoder = motor.CompiledConsolidatedSmoothMotor(motor.make_preview(model))
    assert np.array_equal(decoder.raw_mean(np.zeros(134), 0), np.ones(12) * 99)


def test_stale_parent_binding_rejected_even_after_resealing(model):
    model["base_model"]["frozen_parent"]["model_hash"] = "sha256:" + "f" * 64
    reseal(model)
    with pytest.raises(ValueError):
        motor.validate_model(model)


def test_unbound_experience_mapping_rejected(model):
    model["consolidation_manifest"]["consolidation_mapping"]["sample_to_memory_row"][0] = 100
    manifest = model["consolidation_manifest"]
    manifest["report_hash"] = hash_json({k: v for k, v in manifest.items() if k != "report_hash"})
    reseal(model)
    with pytest.raises(ValueError):
        motor.validate_model(model)


def test_policy_cannot_change_its_embedded_model_without_new_commitment(model):
    policy = motor.make_preview(model)
    policy["step_motor_proof"]["model"] = copy.deepcopy(model)
    policy["step_motor_proof"]["model"]["model_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        motor.CompiledConsolidatedSmoothMotor(policy)


@pytest.mark.parametrize("flag", ["hardware_authorized", "promotion_authorized"])
def test_consolidation_never_grants_authority(model, flag):
    model[flag] = True
    reseal(model)
    with pytest.raises(ValueError):
        motor.validate_model(model)
