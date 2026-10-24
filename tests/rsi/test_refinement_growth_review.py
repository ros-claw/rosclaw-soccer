import copy

import pytest

from rosclaw_soccer.rsi.refinement_growth_review import paired_refinement_growth


def rows():
    return [
        dict(
            course=dict(state_id=s, task=t, assisted=False, context=(1, 4, 20, 21)[s // 10]),
            outcome=dict(
                whole_body_healthy=True,
                passer_full100NNrecovery=True,
                passer_clean_foot=True,
                B_legal_functional_receive_before_deadline=s < 20,
                B_excluded_contact=False,
                combined_precision_and_functional_receive=False,
                incoming=dict(A_first_foot_strike_substep=10, B_first_ball_contact_substep=20),
            ),
        )
        for s in range(40)
        for t in (0, 1)
    ]


def review(old, new, achieved=None):
    return paired_refinement_growth(
        old, new, required_old_contexts={1, 4}, achieved_old_contexts=achieved or {1, 4}
    )


def test_actual_new_contact_growth_and_original_protection_are_separate():
    old = rows()
    new = copy.deepcopy(old)
    new[40]["outcome"]["B_legal_functional_receive_before_deadline"] = True
    result = review(old, new)
    assert result["parent"]["controlled"] == 40
    assert result["candidate"]["controlled"] == 41
    assert result["rescues"] == [[20, 0]]
    assert result["eligible_for_new_sealed_Fresh_research"] is True
    assert review(old, new, {1})["eligible_for_new_sealed_Fresh_research"] is False
    assert result["promotion_authorized"] is False


@pytest.mark.parametrize("fault", ["regression", "nonfoot", "unhealthy", "no_B_contact"])
def test_new_growth_cannot_hide_retention_or_contact_failures(fault):
    old = rows()
    new = copy.deepcopy(old)
    new[40]["outcome"]["B_legal_functional_receive_before_deadline"] = True
    if fault == "regression":
        new[0]["outcome"]["B_legal_functional_receive_before_deadline"] = False
    elif fault == "nonfoot":
        new[42]["outcome"]["passer_clean_foot"] = False
    elif fault == "unhealthy":
        new[42]["outcome"]["whole_body_healthy"] = False
    else:
        new[40]["outcome"]["incoming"]["B_first_ball_contact_substep"] = None
        new[40]["outcome"]["combined_precision_and_functional_receive"] = True
    result = review(old, new)
    assert result["eligible_for_new_sealed_Fresh_research"] is False
    if fault == "no_B_contact":
        assert result["candidate"]["raw_strict"] == 1
        assert result["candidate"]["strict_with_actual_controlled_contact"] == 0


@pytest.mark.parametrize("fault", ["missing", "duplicate", "different_target", "teacher"])
def test_all_whole_paired_public_tasks_are_required(fault):
    old = rows()
    new = copy.deepcopy(old)
    if fault == "missing":
        new.pop()
    elif fault == "duplicate":
        new[1] = copy.deepcopy(new[0])
    elif fault == "different_target":
        new[0]["course"]["target"] = [1, 2, 3]
    else:
        new[0]["course"]["assisted"] = True
    with pytest.raises(ValueError):
        review(old, new)


def test_existing_unsafe_parent_contact_does_not_become_permitted_baseline():
    old = rows()
    old[42]["outcome"]["passer_clean_foot"] = False
    new = copy.deepcopy(old)
    new[40]["outcome"]["B_legal_functional_receive_before_deadline"] = True
    result = review(old, new)
    assert result["paired_net_change"] == 1
    assert result["new_nonfoot"] == []
    assert result["eligible_for_new_sealed_Fresh_research"] is False
