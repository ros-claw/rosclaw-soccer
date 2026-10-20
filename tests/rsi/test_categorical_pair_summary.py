import pytest

from rosclaw_soccer.rsi.categorical_pair_summary import summarize_categorical_pairs


def rows(rescues=6, nonfoot=False):
    return [
        dict(
            course=dict(course_index=i, stratum=f"s{i // 16}"),
            variants=[
                dict(
                    variant=name,
                    causal_qualified=name == "seed17" and i < rescues,
                    score=dict(
                        qualified=name == "seed17" and i < rescues,
                        minimum_pelvis_z_m=0.75,
                        minimum_upright=0.99,
                        maximum_nonfoot_force_n=2.0
                        if name == "seed17" and nonfoot and i == 0
                        else 0.0,
                    ),
                    prefix_diagnostics=dict(
                        prefix_physical_safe=True, post_boundary_control_can_cause_first_touch=True
                    ),
                )
                for name in ("anchor", "seed17", "seed29", "seed43")
            ],
        )
        for i in range(96)
    ]


def test_six_discordant_rescues_control_three_comparisons_not_promotion():
    summary = summarize_categorical_pairs(rows())
    s = summary["comparisons"]["seed17"]
    assert s["paired_one_sided_sign_p"] == 1 / 64
    assert s["bonferroni_three_comparison_p"] == 3 / 64
    assert s["necessary_generalization_evidence"]
    assert not summary["promotion_authorized"]
    assert not summary["original_far_or_private_fresh_pass"]
    assert not summarize_categorical_pairs(rows(5))["comparisons"]["seed17"][
        "necessary_generalization_evidence"
    ]
    assert not summarize_categorical_pairs(rows(nonfoot=True))["comparisons"]["seed17"][
        "necessary_generalization_evidence"
    ]
    assert summary["comparisons"]["seed29"]["paired_one_sided_sign_p"] == 1


@pytest.mark.parametrize("bad", ["missing", "duplicate", "flag", "nan", "stratum"])
def test_incomplete_or_bad_pairs_rejected(bad):
    data = rows()
    if bad == "missing":
        data.pop()
    elif bad == "duplicate":
        data[0] = data[1]
    elif bad == "flag":
        data[0]["variants"][1]["causal_qualified"] = 1
    elif bad == "nan":
        data[0]["variants"][1]["score"]["minimum_upright"] = float("nan")
    else:
        data[0]["course"]["stratum"] = "different"
    with pytest.raises(ValueError):
        summarize_categorical_pairs(data)
