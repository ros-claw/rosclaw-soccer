import copy

import pytest

from scripts.rsi_admit_relay_balance_selector import assess


def evidence():
    choice = dict(
        action=1,
        parent_success=False,
        chosen_success=True,
        parent_failure=False,
        chosen_failure=False,
    )
    rows = [
        dict(context=1, seed=s, heldout_decisions=1, choices=[copy.copy(choice)])
        for s in (17, 43, 71)
    ]
    flags = dict(
        physics_evaluated_as_controller=False,
        safety_certificate=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    source = dict(
        status="RELAY_BALANCE_OFFLINE_SELECTOR_HOLDOUT_NOT_PHYSICS",
        dataset_hash="data",
        contexts=[1],
        rows=rows,
        **flags,
    )
    audit = dict(
        status="RELAY_BALANCE_SELECTOR_NUMPY_REPLAY_NOT_CONTROL",
        dataset_hash="data",
        independent_numpy_inference=True,
        rows=[
            dict(context=1, seed=s, heldout_decisions=1, choices_replayed=True)
            for s in (17, 43, 71)
        ],
        **flags,
    )
    return source, audit


def test_all_seeds_must_show_rescue_without_regression():
    source, audit = evidence()
    assert assess(source, audit)["status"] == "SELECTOR_RESEARCH_PHYSICS_ELIGIBLE"
    source["rows"][1]["choices"][0].update(parent_failure=False, chosen_failure=True)
    result = assess(source, audit)
    assert result["status"] == "SELECTOR_RESEARCH_PHYSICS_REJECTED"
    assert result["metrics"][43]["failure_regressions"] == 1
    assert result["hardware_authorized"] is False


def test_incomplete_or_wrong_evidence_is_rejected():
    source, audit = evidence()
    audit["rows"].pop()
    with pytest.raises(ValueError):
        assess(source, audit)
    source, audit = evidence()
    source["rows"][0]["choices"][0]["parent_success"] = 1
    with pytest.raises(ValueError):
        assess(source, audit)
    source, audit = evidence()
    audit["dataset_hash"] = "different"
    with pytest.raises(ValueError):
        assess(source, audit)
