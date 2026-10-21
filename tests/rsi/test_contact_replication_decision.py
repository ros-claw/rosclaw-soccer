import copy

import pytest

from rosclaw_soccer.rsi.contact_replication_decision import ball_outcome, decide


def rows():
    return [
        dict(
            seed=seed,
            task_set=task,
            initial=dict(clean=25, stable=64, forbidden=23, ball_outcome=3),
            final=dict(clean=26, stable=64, forbidden=23, ball_outcome=3),
        )
        for seed in (7602, 7605, 7606)
        for task in ("train64", "dev64")
    ]


def test_event_gain_is_not_ball_outcome_gain():
    result = decide(rows())
    assert result["decision"].startswith("STOP_UNCHANGED")
    assert not result["promotion_authorized"]


def test_all_replications_required_and_safety_has_priority():
    data = rows()
    data[-1]["final"]["ball_outcome"] = 4
    assert decide(data)["decision"].startswith("STOP_UNCHANGED")
    data[3]["final"]["ball_outcome"] = 4
    assert decide(data)["decision"] == "REQUIRES_ORIGINAL_FORMAL52_AND_SEALED_FRESH"
    data[0]["final"]["forbidden"] += 1
    assert decide(data)["decision"] == "REJECT_SAFETY_REGRESSION"
    with pytest.raises(ValueError):
        decide(data[:-1])
    with pytest.raises(ValueError):
        decide(data[:-1] + [copy.deepcopy(data[0])])


def test_censored_or_forbidden_is_not_success():
    row = dict(
        stable=True,
        forbidden=False,
        postcontact=dict(
            post_contact_quality_observed=True,
            goal_progress_1p2s_m=1.0,
            goal_lateral_1p2s_m=0.3,
        ),
    )
    assert ball_outcome(row)
    row["forbidden"] = True
    assert not ball_outcome(row)
    row["forbidden"] = False
    row["postcontact"]["post_contact_quality_observed"] = False
    assert not ball_outcome(row)
