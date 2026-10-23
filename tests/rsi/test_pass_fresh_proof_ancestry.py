import copy

import pytest

from scripts.rsi_seal_consolidated_pass_fresh import validate_pass_ancestry


def proofs():
    # Real own-pass schema intentionally has no copied checkpoint_kind/hash.
    own = dict(
        training_hash="training",
        counts=dict(native_steps=10),
        records=[{}] * 80,
        summary=dict(candidate_controlled=41),
        comparison=[{}] * 80,
    )
    source = dict(
        copy.deepcopy(own),
        status="TEAM_PASS_CANDIDATE80_COLLECTED_PENDING_OWN_NATIVE_AND_RETENTION",
        checkpoint_kind="consolidated-adapter-v1",
    )
    declared = dict(
        training_hash="training",
        checkpoint_kind="consolidated-adapter-v1",
        cpu_audit_hash="CPU",
        teacher_enabled=False,
        B_policy_frozen=True,
    )
    trained = dict(checkpoint_kind="consolidated-adapter-v1")
    return own, source, declared, trained


def test_exact_real_own_schema_resolves_kind_via_bound_producer_and_declaration():
    validate_pass_ancestry(*proofs(), "CPU")


@pytest.mark.parametrize("fault", ["model", "kind", "scope", "CPU", "teacher", "counts", "summary"])
def test_source_binding_failure_never_defaults_the_new_model_kind(fault):
    own, source, declared, trained = proofs()
    if fault == "model":
        source["training_hash"] = "other"
    elif fault == "kind":
        declared["checkpoint_kind"] = "other"
    elif fault == "scope":
        own["records"].pop()
    elif fault == "CPU":
        declared["cpu_audit_hash"] = "other"
    elif fault == "teacher":
        declared["teacher_enabled"] = True
    elif fault == "counts":
        source["counts"]["native_steps"] = 9
    elif fault == "summary":
        source["summary"]["candidate_controlled"] = 99
    with pytest.raises(ValueError):
        validate_pass_ancestry(own, source, declared, trained, "CPU")
