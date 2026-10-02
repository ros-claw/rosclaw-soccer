"""Synthetic contract and learning tests; not physical skill evidence."""

import copy

import numpy as np
import pytest
from rosclaw.growth.anchor_consolidation import consolidate_unique
from rosclaw.growth.anchor_output_memory import AnchorOutputMemory

from rosclaw_soccer.rsi.consolidated_smooth_motor import make_model as make_baseline
from rosclaw_soccer.rsi.current_memory_learning import fit_update
from rosclaw_soccer.rsi.current_memory_motor import (
    CompiledCurrentMemoryMotor,
    initial_model,
    make_preview,
    validate_model,
)
from rosclaw_soccer.rsi.smooth_memory_motor import CompiledSmoothMemoryMotor
from rosclaw_soccer.rsi.smooth_memory_motor import make_preview as old_preview
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_correlated_gradient_numerical_parity import conditional_fixture_batch
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.fixture
def current(smooth_parent):  # noqa: F811
    inherited = smooth_parent["frozen_parent"]["output_memory"]
    old = AnchorOutputMemory.from_dict(inherited)
    decoder = CompiledSmoothMemoryMotor(old_preview(smooth_parent))
    extra = np.full(134, 0.37)
    context = np.concatenate((decoder.features(extra)[:134], [1.0]))
    evidence = dict(
        schema="soccer.rsi.current_parent_success_consolidation_evidence.v1",
        parent_model_hash=smooth_parent["parent_model_hash"],
        encoder_hash=old.encoder_hash,
        inherited_memory_hash=inherited["memory_hash"],
        bank_summary_hash="synthetic-bank",
        bank_review_hash="synthetic-review",
        selection="ALL_QUALIFIED_CURRENT_PARENT_SUCCESSES_IN_SEALED_ORDER",
        records=[dict(seed=42, lane=0, frames=270)],
        source_hash="synthetic-source",
        consolidation_source_hash="synthetic-source",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    memory, mapping = consolidate_unique(
        old,
        np.tile(context, (270, 1)),
        np.tile(decoder.raw_mean(extra, 1), (270, 1)),
        parent_policy_hash=smooth_parent["parent_model_hash"],
        evidence_hash=hash_json(evidence),
    )
    serialized = memory.to_dict()
    manifest = dict(
        **evidence,
        consolidation_mapping=mapping,
        memory_hash=serialized["memory_hash"],
        inherited_frames=len(inherited["observations"]),
        recorded_frames=len(serialized["observations"]),
        added_frames=270,
        independent_contexts=1,
        existing_success_reports_reconstructed=1,
        actual_physical_executions_added=0,
        exact_current_parent_output_reload=True,
        physical_policy_execution_qualified=False,
    )
    manifest["report_hash"] = hash_json(manifest)
    return initial_model(make_baseline(smooth_parent, serialized, manifest)), extra


def test_zero_addition_matches_behavior_globally_and_new_anchor_is_protected(
    current,
    smooth_parent,  # noqa: F811
):  # noqa: F811
    proposal, extra = current
    new = CompiledCurrentMemoryMotor(make_preview(proposal))
    old = CompiledSmoothMemoryMotor(old_preview(smooth_parent))
    for phase in range(3):
        for x in (extra, np.zeros(134), np.ones(134), np.full(134, -0.19)):
            assert np.array_equal(new.raw_mean(x, phase), old.raw_mean(x, phase))
    context = np.concatenate((new.features(extra)[:134], [1.0]))
    assert old._guard.gate(context) > 0
    assert new._guard.gate(context) == 0
    assert new._cap == 0.2


@pytest.mark.parametrize(
    "key,value",
    [
        ("raw_residual_cap", 0.3),
        ("raw_residual_cap", True),
        ("learning_rate", 1e-2),
        ("guard_bandwidth", 1),
        ("hardware_authorized", True),
        ("promotion_authorized", True),
    ],
)
def test_resealed_contract_expansion_rejected(current, key, value):
    proposal = copy.deepcopy(current[0])
    proposal[key] = value
    proposal.pop("model_hash")
    proposal["model_hash"] = hash_json(proposal)
    with pytest.raises(ValueError):
        validate_model(proposal)


def test_optimizer_protects_current_memory_without_post_training_blend(current, smooth_parent):  # noqa: F811
    proposal, extra = current
    data = conditional_fixture_batch(smooth_parent)
    learned = fit_update(proposal, data, batch_hash="sha256:" + "b" * 64)
    new, old = (
        CompiledCurrentMemoryMotor(make_preview(learned)),
        CompiledCurrentMemoryMotor(make_preview(proposal)),
    )
    assert np.array_equal(new.raw_mean(extra, 1), old.raw_mean(extra, 1))
    assert learned["baseline"] == proposal["baseline"]
    assert learned["learning_receipt"]["completed_optimizer_steps"] > 0
    assert learned["learning_receipt"]["residual_cap"] == 0.2
    assert learned["learning_receipt"]["learning_rate"] == 4e-4
    assert learned["learning_receipt"]["physical_batch_verified"] is False
    assert learned["learning_receipt"]["promotion_authorized"] is False
    assert learned["residual_layers"] != proposal["residual_layers"]
    from scripts.rsi_collect_protected_phase_bank_validation import validate_bank_models

    validate_bank_models(learned, proposal["baseline"]["base_model"]["frozen_parent"])
    for key, value in (
        ("protected_memory_hash", "sha256:" + "e" * 64),
        ("behavior_model_hash", "sha256:" + "e" * 64),
        ("protected_anchor_contexts", 999),
    ):
        forged = copy.deepcopy(learned)
        forged["learning_receipt"][key] = value
        forged.pop("model_hash")
        forged["model_hash"] = hash_json(forged)
        with pytest.raises(ValueError, match="receipt"):
            validate_model(forged)
    with pytest.raises(ValueError, match="zero-addition"):
        fit_update(learned, data, batch_hash="sha256:" + "b" * 64)


def test_counterexample_control_requires_full_array_identity():
    from scripts.rsi_retest_current_memory_counterexample import check_trace_arrays

    before = {"qpos": np.arange(4, dtype=np.float64), "ball": np.zeros((3, 3))}
    check_trace_arrays(before, copy.deepcopy(before))
    for replacement in (
        {"qpos": before["qpos"]},
        {**before, "qpos": before["qpos"].astype(np.float32)},
        {**before, "ball": np.ones((3, 3))},
    ):
        with pytest.raises(ValueError, match="reproduce every"):
            check_trace_arrays(before, replacement)
