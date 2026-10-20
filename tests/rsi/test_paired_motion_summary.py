import copy

import pytest

from rosclaw_soccer.rsi.paired_motion_summary import summarize_pairs


def pair(index, b, p):
    return dict(
        course=dict(course_index=index, stratum="test"),
        rescue=not b and p,
        regression=b and not p,
        variants=[
            dict(
                variant=name,
                causal_qualified=passed,
                score=dict(
                    qualified=passed,
                    minimum_pelvis_z_m=0.75,
                    minimum_upright=0.99,
                    maximum_nonfoot_force_n=0.0,
                ),
                prefix_diagnostics=dict(
                    prefix_physical_safe=True, post_boundary_control_can_cause_first_touch=True
                ),
                portfolio_audit=dict(selected_owner=0),
            )
            for name, passed in (("baseline", b), ("portfolio", p))
        ],
    )


def test_no_change_is_not_breakthrough():
    result = summarize_pairs([pair(0, True, True), pair(1, False, False)])
    assert result["paired_one_sided_sign_p"] == 1
    assert result["net_gain"] == 0
    assert result["totals"]["both_success"] == 1
    assert result["totals"]["both_failure"] == 1
    assert result["promotion_authorized"] is False


def test_exact_paired_probability_not_frame_pseudoreplication():
    result = summarize_pairs([pair(i, False, True) for i in range(5)])
    assert result["paired_one_sided_sign_p"] == 1 / 32
    assert result["net_gain"] == 5
    assert result["totals"]["courses"] == 5


def test_regressions_count_against_rescues():
    result = summarize_pairs([pair(0, False, True), pair(1, True, False)])
    assert result["paired_one_sided_sign_p"] == 0.75
    assert result["net_gain"] == 0


@pytest.mark.parametrize("mutation", ["duplicate", "causal", "change", "nonfinite", "order"])
def test_reject_inconsistent_accounting(mutation):
    rows = [pair(0, False, True)]
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "causal":
        rows[0]["variants"][1]["prefix_diagnostics"]["prefix_physical_safe"] = False
    elif mutation == "change":
        rows[0]["rescue"] = False
    elif mutation == "nonfinite":
        rows[0]["variants"][0]["score"]["minimum_upright"] = float("nan")
    else:
        rows[0]["variants"].reverse()
    with pytest.raises(ValueError):
        summarize_pairs(rows)
