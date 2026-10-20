"""Accounting fixtures, not task or independent physics evidence."""

import copy

import pytest

from scripts.rsi_native_context_recovery_summary import summarize_slots


def rows():
    return [
        dict(slot=f"c{c}:v{v}:{a}", status="COMPLETE", native_steps=3000, qualified=False)
        for c in range(4)
        for v in range(4)
        for a in ("baseline", "blend_only", "measured_blend")
    ]


def test_full_accounting_keeps_fault_quality_unobserved():
    data = rows()
    data[0]["qualified"] = True
    data[2].update(status="POLICY_REFERENCE_REJECTED", native_steps=400, qualified=None)
    data[5]["qualified"] = True
    out = summarize_slots(data)
    assert out["total_integrated_native_steps"] == 141400
    assert out["by_arm"]["measured_blend"] == dict(
        attempted_slots=16,
        full_h300_cases=15,
        guard_rejected_quality_unobserved=1,
        producer_claimed_qualified=1,
        native_steps=45400,
    )
    pair = out["paired_comparisons"]["measured_blend"]
    assert pair == dict(
        mutually_complete_pairs=15,
        observed_rescues=1,
        observed_regressions=0,
        baseline_success_with_changed_guard_rejection=1,
    )
    assert out["producer_claims_only"] and not out["independent_complete_physics_review"]
    assert not out["promotion_authorized"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered", "foreign"])
def test_all_exact_ordered_slots_required(mutation):
    data = rows()
    if mutation == "missing":
        data.pop()
    elif mutation == "duplicate":
        data[-1] = copy.deepcopy(data[0])
    elif mutation == "reordered":
        data[0], data[1] = data[1], data[0]
    else:
        data[0]["slot"] = "another_bank"
    with pytest.raises(ValueError, match="original48"):
        summarize_slots(data)


@pytest.mark.parametrize(
    "patch",
    [
        dict(status="NOT_EXECUTED"),
        dict(native_steps=2999),
        dict(qualified=None),
        dict(status="POLICY_REFERENCE_REJECTED", native_steps=400, qualified=False),
        dict(status="POLICY_REFERENCE_REJECTED", native_steps=401, qualified=None),
        dict(status="POLICY_REFERENCE_REJECTED", native_steps=3000, qualified=None),
    ],
)
def test_missing_or_unobserved_quality_not_fabricated(patch):
    data = rows()
    data[0].update(patch)
    with pytest.raises(ValueError):
        summarize_slots(data)
