import copy

import pytest

from rosclaw_soccer.rsi.paid_skill_frontier import compare_skill_frontier


def fixture():
    report = dict(
        bank="train",
        worlds=[dict(case=i, stable=True, quality_by_kick=[True, True, i < 13]) for i in range(16)],
    )
    return report, copy.deepcopy(report)


def test_equal_aggregate_count_does_not_hide_lost_acquired_skill():
    parent, child = fixture()
    child["worlds"][1]["quality_by_kick"][2] = False
    child["worlds"][14]["quality_by_kick"][2] = True
    result = compare_skill_frontier(parent, child)
    assert result["reference_counts"] == result["candidate_counts"]
    assert result["lost_by_kick"][2] == [1] and result["gained_by_kick"][2] == [14]
    assert not result["no_regression"] and not result["promotion_authorized"]


def test_early_body_failure_retains_missing_later_kicks_as_failures():
    parent, child = fixture()
    child["worlds"][0].update(stable=False, quality_by_kick=[True])
    result = compare_skill_frontier(parent, child)
    assert result["lost_by_kick"] == [[], [0], [0]]
    assert result["body_regressions"] == [0]


def test_genuine_frontier_expansion_keeps_old_successes():
    parent, child = fixture()
    for i in (13, 14):
        child["worlds"][i]["quality_by_kick"][2] = True
    result = compare_skill_frontier(parent, child)
    assert result["no_regression"] and result["gained_by_kick"] == [[], [], [13, 14]]


@pytest.mark.parametrize(
    "fault", ["bank", "missing", "duplicate", "string_boolean", "missing_body"]
)
def test_weak_or_unmatched_evidence_is_not_a_retention_result(fault):
    parent, child = fixture()
    if fault == "bank":
        child["bank"] = "dev"
    elif fault == "missing":
        child["worlds"].pop()
    elif fault == "duplicate":
        child["worlds"][1]["case"] = 0
    elif fault == "string_boolean":
        child["worlds"][0]["quality_by_kick"][0] = "false"
    else:
        del child["worlds"][0]["stable"]
    with pytest.raises(ValueError):
        compare_skill_frontier(parent, child)
