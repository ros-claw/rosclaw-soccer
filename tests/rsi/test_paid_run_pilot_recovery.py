import copy

import pytest

from rosclaw_soccer.rsi.paid_run_pilot_recovery import contact_quality


def result():
    return dict(
        body_failed=False,
        forbidden=False,
        postcontact=dict(
            correct_foot_first=True,
            post_contact_quality_observed=True,
            goal_progress_1p2s_m=1.0,
            goal_lateral_1p2s_m=0.75,
            other_foot_contact_in_window=False,
            maximum_pelvis_retreat_after_contact_m=0.05,
        ),
    )


def test_quality_has_no_search_reward_dependency_and_keeps_original_thresholds():
    r = result()
    assert contact_quality(r)
    for reward in (-100.0, 0.0, 999.0):
        variant = copy.deepcopy(r)
        variant["terminal_reward"] = reward
        assert contact_quality(variant)


@pytest.mark.parametrize(
    "kind",
    [
        "no_contact",
        "censored",
        "unsafe",
        "body_failed",
        "wrong_foot",
        "poor_progress",
        "lateral",
        "retreat",
    ],
)
def test_ordinary_failed_or_censored_outcomes_do_not_crash(kind):
    r = result()
    p = r["postcontact"]
    if kind == "no_contact":
        p.update(
            correct_foot_first=False, post_contact_quality_observed=False, goal_progress_1p2s_m=None
        )
    elif kind == "censored":
        p.update(post_contact_quality_observed=False, goal_progress_1p2s_m=None)
    elif kind == "unsafe":
        r["forbidden"] = True
    elif kind == "body_failed":
        r["body_failed"] = True
    elif kind == "wrong_foot":
        p["correct_foot_first"] = False
    elif kind == "poor_progress":
        p["goal_progress_1p2s_m"] = 0.99
    elif kind == "lateral":
        p["goal_lateral_1p2s_m"] = 0.751
    else:
        p["maximum_pelvis_retreat_after_contact_m"] = 0.051
    assert not contact_quality(r)
