import copy

import pytest

from rosclaw_soccer.rsi.native_role_learning_review import review_native_role_learning
from rosclaw_soccer.training.native_team_courses import native_team_course_bank


def reports():
    metric = dict(
        role_return=[0.0] * 8,
        actual_foot_contact_steps=[0, 0, 0, 10, 0, 0, 0, 0],
        close_recent_comotion_controls=[0] * 8,
    )
    return [
        dict(
            seed=s,
            status="NATIVE_ROLE_TD3_ONE_GENERATION_PHYSICALLY_MEASURED_NOT_PROMOTED",
            all5_cases_measured=True,
            goal_complete=False,
            promotion_authorized=False,
            hardware_authorized=False,
            results=[
                dict(
                    course=c.course_id,
                    candidate_healthy=True,
                    parent_metrics=copy.deepcopy(metric),
                    candidate_metrics=copy.deepcopy(metric),
                )
                for c in native_team_course_bank()
            ],
        )
        for s in (926061, 926062, 926063)
    ]


def test_identical_or_positive_proxy_never_promotes():
    data = reports()
    for report in data:
        report["results"][0]["candidate_metrics"]["close_recent_comotion_controls"][3] = 1
    before = copy.deepcopy(data)
    result = review_native_role_learning(data)
    assert result["every_seed_has_positive_comotion_proxy"]
    assert not result["physical_possession_qualified"]
    assert not result["promotion_authorized"]
    assert not result["goal_complete"]
    assert data == before


def test_all_role_failures_retained_not_best_seed():
    data = reports()
    data[1]["results"][2]["candidate_healthy"] = False
    data[2]["results"][4]["candidate_metrics"]["actual_foot_contact_steps"][3] = 0
    result = review_native_role_learning(data)
    assert not result["all_candidates_healthy"]
    assert result["lost_parent_contact_cases"] == [(926063, "incoming-west", 3)]
    assert len(result["rows"]) == 15


@pytest.mark.parametrize("fault", ["missing_seed", "missing_case", "nonfinite", "missing_role"])
def test_incomplete_or_invalid_bank_rejected(fault):
    data = reports()
    if fault == "missing_seed":
        data.pop()
    elif fault == "missing_case":
        data[0]["results"].pop()
    elif fault == "nonfinite":
        data[0]["results"][0]["candidate_metrics"]["role_return"][0] = float("nan")
    else:
        data[0]["results"][0]["candidate_metrics"]["actual_foot_contact_steps"].pop()
    with pytest.raises(ValueError):
        review_native_role_learning(data)
