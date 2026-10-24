import json

import pytest

from rosclaw_soccer.rsi import supported_motor_checkpoint as admission
from rosclaw_soccer.rsi.dual_motor_checkpoint import digest


def test_scope_requires_both_actual_current_motion_and_original_whole_bank(monkeypatch):
    f = dict(
        status="PASS_REFINEMENT_OWN_FEATURES_ZERO_IDENTITY_NOT_PHYSICS",
        bank_hash="bank",
        parent_training_hash="parent",
        records=[dict(rows=375)] * 61 + [dict(rows=366)],
        own_feature_decisions=23241,
    )
    old = dict(
        status="CURRENT_PARENT_RETENTION_OWN_FEATURES_MATCH_ACTUAL_PHYSICS_NOT_NEW_LEARNING",
        feature_hash="features",
        parent_training_hash="parent",
        records=[dict(rows=330)] * 16 + [dict(rows=456)],
        own_actor_decisions=5736,
    )
    monkeypatch.setattr(
        admission, "support_receipt", lambda r, h: (None, {"features": f, "motion": old}[h])
    )
    r = dict(
        feature_hash="features",
        retention_feature_hash="motion",
        bank_hash="bank",
        parent_training_hash="parent",
    )
    assert admission.support_cpu_scope(r) == 28977
    old["feature_hash"] = "different"
    with pytest.raises(ValueError, match="scope"):
        admission.support_cpu_scope(r)


@pytest.mark.parametrize("fault", ["optimizer", "promotion", "descriptor", "escape", "declaration"])
def test_bundle_rejects_new_authority_or_fake_update_before_opening_models(tmp_path, fault):
    ids = {k: "sha256:" + str(i).zfill(64) for i, k in enumerate(admission.IDENTITIES)}
    ids["parent_lineage_hash"] = "parent"
    cp = tmp_path / "supported-motor.json"
    descriptor = dict(schema=admission.KIND, **ids)
    if fault == "descriptor":
        descriptor.pop("gate_audit_hash")
    cp.write_text(json.dumps(descriptor))
    declared = dict(
        **ids,
        source_head="head",
        input_file_hashes={},
        config=dict(
            seed=115025, new_optimizer_updates=0, reused_head_updates=2048, support_schema=1
        ),
        checkpoint_path=str(cp),
        checkpoint_hash=digest(cp),
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
    )
    if fault == "escape":
        declared["checkpoint_path"] = str(tmp_path.parent / "outside.json")
    dp = tmp_path / "declaration.json"
    dp.write_text(json.dumps(declared))
    report = dict(
        **declared,
        status=admission.STATUS,
        checkpoint_kind=admission.KIND,
        declaration_hash=digest(dp),
        checkpoint_format="json.reference_bundle.v1",
        new_optimizer_updates=0,
        physical_steps_executed=0,
        old_retention_proven=False,
    )
    if fault == "optimizer":
        report["new_optimizer_updates"] = 1
    elif fault == "promotion":
        report["promotion_authorized"] = True
    elif fault == "declaration":
        report["declaration_hash"] = "changed"
    rp = tmp_path / "complete.json"
    rp.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        admission.load_supported_motor(
            None, parent_lineage_hash="parent", root=tmp_path, expected_receipt_hash=digest(rp)
        )
