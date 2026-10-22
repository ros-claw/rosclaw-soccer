import copy

import pytest

from rosclaw_soccer.rsi.motor_exploration_comparison import compare_motor_profiles, physical_readout
from rosclaw_soccer.rsi.motor_exploration_curriculum import motor_curriculum


def rows():
    return [
        dict(
            context=c,
            seed=s,
            healthy_quality=False,
            assisted=False,
            rejection=None,
            result=dict(
                body_failed=False,
                forbidden=True,
                postcontact=dict(
                    correct_foot_first=True,
                    post_contact_quality_observed=True,
                    goal_progress_1p2s_m=1.0,
                    goal_lateral_1p2s_m=0.2,
                    maximum_pelvis_retreat_after_contact_m=0.01,
                ),
            ),
        )
        for c, s in motor_curriculum("retention-run64").shard_pairs(4, 0, 1)
    ]


def test_no_discovery_pivots_not_more_blind_ppo():
    a = rows()
    r = compare_motor_profiles(a, copy.deepcopy(a), generation=4)
    assert r["next_research_action"].startswith("PIVOT")
    assert r["distinct_task_contexts"] == 4
    assert r["paired_worlds"] == 64 and r["total_executions"] == 128
    assert not r["new_policy_learning_proven"]


def test_paired_rescue_regression_not_total_success_alone():
    a = rows()
    b = copy.deepcopy(a)
    a[0]["healthy_quality"] = True
    a[0]["result"]["forbidden"] = False
    b[1]["healthy_quality"] = True
    b[1]["result"]["forbidden"] = False
    r = compare_motor_profiles(a, b, generation=4)
    assert r["contexts"]["4"]["rescues"] == r["contexts"]["4"]["regressions"] == 1
    assert r["next_research_action"].startswith("PIVOT")
    b[2]["healthy_quality"] = True
    b[2]["result"]["forbidden"] = False
    r = compare_motor_profiles(a, b, generation=4)
    assert r["next_research_action"].startswith("ONE_BOUNDED")
    b[-1]["result"]["body_failed"] = True
    assert compare_motor_profiles(a, b, generation=4)["next_research_action"].startswith("PIVOT")


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "seed", "assisted", "rejected"])
def test_cannot_hide_failed_worlds_or_unpair_courses(mutation):
    a, b = rows(), rows()
    if mutation == "missing":
        b.pop()
    elif mutation == "duplicate":
        b[-1] = copy.deepcopy(b[0])
    elif mutation == "seed":
        b[-1]["seed"] += 1000
    elif mutation == "assisted":
        b[-1]["assisted"] = True
    else:
        b[-1]["rejection"] = dict(executed=False)
    with pytest.raises(ValueError):
        compare_motor_profiles(a, b, generation=4)


def test_censored_windows_are_not_synthetic_measurements():
    a = rows()[:1]
    a[0]["result"]["postcontact"].update(
        post_contact_quality_observed=False,
        goal_progress_1p2s_m=None,
        goal_lateral_1p2s_m=None,
        maximum_pelvis_retreat_after_contact_m=None,
    )
    assert physical_readout(a)["goal_progress_1p2s_m_median"] is None
    a[0]["result"]["postcontact"]["post_contact_quality_observed"] = True
    with pytest.raises(ValueError):
        physical_readout(a)
