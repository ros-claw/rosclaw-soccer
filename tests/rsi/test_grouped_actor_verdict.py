"""Fixture gates only, never physical football results."""

import pytest

from rosclaw_soccer.rsi.grouped_actor_verdict import grouped_actor_verdict


def rows(passed, nonfoot=0):
    bad = set(sorted(set(range(40)) - passed)[:nonfoot])
    return [
        dict(
            course_index=i,
            split="validation" if i == 20 else "train",
            score=dict(
                qualified=i in passed,
                minimum_pelvis_z_m=0.75,
                minimum_upright=0.99,
                maximum_nonfoot_force_n=2.0 if i in bad else 0.0,
            ),
            prefix_diagnostics=dict(
                prefix_physical_safe=True, post_boundary_control_can_cause_first_touch=True
            ),
        )
        for i in range(40)
    ]


def test_more_successes_still_need_retention_and_guardrails():
    result = grouped_actor_verdict(rows({0, 10, 15, 16, 20}, nonfoot=18))
    assert result["necessary_research_growth"] is True
    assert result["promotion_authorized"] is False
    assert result["original_far_or_private_fresh_pass"] is False
    assert result["validation_success"] == 1
    assert not grouped_actor_verdict(rows({0, 10, 15, 16, 20}, nonfoot=19))[
        "necessary_research_growth"
    ]
    assert not grouped_actor_verdict(rows({10, 15, 16, 20, 24}))["necessary_research_growth"]
    assert not grouped_actor_verdict(rows({0, 10, 15, 16}))["necessary_research_growth"]


@pytest.mark.parametrize("bad", ["duplicate", "missing", "nonfinite", "bool"])
def test_incomplete_or_untyped_exam_never_passes(bad):
    data = rows({0, 10, 15, 16, 20})
    if bad == "duplicate":
        data[1] = data[0]
    elif bad == "missing":
        data.pop()
    elif bad == "nonfinite":
        data[1]["score"]["minimum_upright"] = float("nan")
    else:
        data[1]["score"]["qualified"] = 1
    with pytest.raises(ValueError):
        grouped_actor_verdict(data)
