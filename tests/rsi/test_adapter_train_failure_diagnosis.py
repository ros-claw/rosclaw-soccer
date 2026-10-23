"""Synthetic component accounting; no physical-success evidence."""

from copy import deepcopy

import pytest

from scripts.rsi_diagnose_adapter_train_failures import diagnose


def fixture():
    records, current = [], []
    for i in range(80):
        c = dict(state_id=i // 2, task=i % 2, context=(1, 4, 20, 21)[i // 20])
        outcome = dict(
            whole_body_healthy=True,
            passer_full100NNrecovery=True,
            passer_clean_foot=True,
            B_legal_functional_receive_before_deadline=True,
            B_excluded_contact=False,
            incoming=dict(A_first_foot_strike_substep=100, B_first_ball_contact_substep=200),
            precise_moderate_delivery=False,
        )
        records.append(dict(role="new_unassisted", source_course=c, measured_outcome=outcome))
        current.append(dict(course=deepcopy(c), outcome=deepcopy(outcome)))
    return records, current, dict(teacher_indices=[0], baseline_pass_indices=[1])


def test_safe_old_failure_is_counted_without_becoming_a_success():
    old, new, plan = fixture()
    for o in (old[2]["measured_outcome"], new[2]["outcome"]):
        o["B_legal_functional_receive_before_deadline"] = False
    r = diagnose(old, new, plan)["1"]
    assert r["baseline_controlled"] == r["candidate_controlled"] == 19
    assert r["baseline_clean_without_selected_plan"] == 18
    assert r["baseline_B_contact"] == 20 and r["baseline_B_control"] == 19
    assert r["candidate_precise_moderate_delivery"] == 0


def test_new_nonfoot_is_split_by_preexisting_fit_coverage_not_excused():
    old, new, plan = fixture()
    new[0]["outcome"]["passer_clean_foot"] = False
    new[2]["outcome"]["passer_clean_foot"] = False
    r = diagnose(old, new, plan)["1"]
    assert r["new_nonfoot"] == r["regressions"] == 2
    assert r["new_nonfoot_with_selected_plan"] == 1
    assert r["new_nonfoot_without_selected_plan"] == 1
    assert r["rescues"] == 0


@pytest.mark.parametrize("fault", ["missing", "duplicate", "different_goal", "unbalanced"])
def test_exact_matched_whole_bank_required(fault):
    old, new, plan = fixture()
    if fault == "missing":
        new.pop()
    elif fault == "duplicate":
        new[1] = deepcopy(new[0])
    elif fault == "different_goal":
        new[0]["course"]["target"] = [99]
    else:
        new[0]["course"]["context"] = 4
        old[0]["source_course"]["context"] = 4
    with pytest.raises(ValueError):
        diagnose(old, new, plan)


def test_no_causal_claim_or_model_update_is_made_by_component_counter():
    old, new, plan = fixture()
    before = deepcopy((old, new, plan))
    result = diagnose(old, new, plan)
    assert (old, new, plan) == before
    assert sum(r["candidate_controlled"] for r in result.values()) == 80
    assert sum(r["new_nonfoot"] for r in result.values()) == 0
