import copy

import pytest
from test_paid_fresh_evidence import fixture

from rosclaw_soccer.rsi.paid_fresh_protocol import paired_fresh_review


def data():
    d, _ = fixture()
    r = [dict(case=i, stable=True, quality_by_kick=[True, True, i < 16]) for i in range(32)]
    return {name: copy.deepcopy(r) for name in ("r2", "parent", "progressive")}, d


def test_new_context_growth_requires_no_old_case_loss_not_just_higher_total():
    arms, d = data()
    for i in range(16, 21):
        arms["progressive"][i]["quality_by_kick"][2] = True
    out = paired_fresh_review(arms, d["cases"], d["evaluation"])
    assert out["preregistered_outcome_conditions_met"] and not out["promotion_authorized"]
    arms["progressive"][0]["quality_by_kick"][2] = False
    out = paired_fresh_review(arms, d["cases"], d["evaluation"])
    assert out["comparisons"]["parent"]["third_net_gain"] == 4
    assert out["comparisons"]["parent"]["lost_by_kick"][2] == [0]
    assert not out["preregistered_outcome_conditions_met"]


def test_unreached_later_kicks_and_body_failure_remain_failures():
    arms, d = data()
    arms["progressive"][0].update(quality_by_kick=[True], stable=False)
    out = paired_fresh_review(arms, d["cases"], d["evaluation"])
    assert out["comparisons"]["r2"]["lost_by_kick"] == [[], [0], [0]]
    assert out["comparisons"]["r2"]["body_regressions"] == [0]
    assert not out["preregistered_outcome_conditions_met"]


def test_uncertainty_uses_eight_source_clusters_not_frames_or_repeats():
    arms, d = data()
    out = paired_fresh_review(arms, d["cases"], d["evaluation"])
    assert out["comparisons"]["r2"]["statistical_clusters"] == 8
    assert out["comparisons"]["r2"]["third_source_cluster_bootstrap95_effect_per_case"] == [0, 0]
    assert out == paired_fresh_review(arms, d["cases"], d["evaluation"])
    del arms["parent"]
    with pytest.raises(ValueError):
        paired_fresh_review(arms, d["cases"], d["evaluation"])
