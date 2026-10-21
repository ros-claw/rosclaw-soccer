import pytest

from rosclaw_soccer.rsi.paid_contact_diagnosis import (
    contact_failure_causes,
    summarize_contact_failures,
)


def row():
    return dict(
        body_failed=False,
        forbidden=False,
        postcontact=dict(
            first_foot_contact_substep=1000,
            correct_foot_first=True,
            post_contact_quality_observed=True,
            other_foot_contact_in_window=False,
            goal_progress_1p2s_m=3.0,
            goal_lateral_1p2s_m=0.2,
            maximum_pelvis_retreat_after_contact_m=0.0,
        ),
    )


def test_zero_quality_with_all_foot_contacts_must_not_be_called_unreachable():
    rows = [row() for _ in range(8)]
    for r in rows:
        r["forbidden"] = True
    out = summarize_contact_failures(rows)
    assert out["quality_pass"] == 0 and out["actual_foot_contacts"] == 8
    assert (
        out["diagnostic_direction"] == "INVESTIGATE_CONTACT_CLEANLINESS_TIMING_AND_BALL_DIRECTION"
    )
    assert not out["score_contract_changed"] and not out["promotion_authorized"]


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("side", "excessive_lateral_error"),
        ("retreat", "excessive_post_contact_retreat"),
        ("progress", "insufficient_forward_progress"),
        ("window", "post_contact_window_unobserved"),
        ("no_foot", "no_foot_contact"),
    ],
)
def test_contact_failure_types_are_separate(fault, expected):
    r = row()
    if fault == "side":
        r["postcontact"]["goal_lateral_1p2s_m"] = 0.751
    elif fault == "retreat":
        r["postcontact"]["maximum_pelvis_retreat_after_contact_m"] = 0.051
    elif fault == "progress":
        r["postcontact"]["goal_progress_1p2s_m"] = 0.999
    else:
        r["postcontact"]["post_contact_quality_observed"] = False
        r["postcontact"]["other_foot_contact_in_window"] = None
        if fault == "no_foot":
            r["postcontact"]["first_foot_contact_substep"] = None
            r["postcontact"]["correct_foot_first"] = False
    assert expected in contact_failure_causes(r)


def test_exact_boundaries_remain_frozen_and_nonfinite_values_are_not_sanitized_into_success():
    r = row()
    r["postcontact"].update(
        goal_progress_1p2s_m=1.0,
        goal_lateral_1p2s_m=0.75,
        maximum_pelvis_retreat_after_contact_m=0.05,
    )
    assert contact_failure_causes(r) == []
    r["postcontact"]["goal_lateral_1p2s_m"] = float("nan")
    with pytest.raises(ValueError):
        contact_failure_causes(r)
