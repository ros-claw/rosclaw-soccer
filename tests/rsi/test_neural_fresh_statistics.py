import copy

import pytest

from rosclaw_soccer.rsi.neural_fresh_statistics import paired_cluster_score


def data():
    protocol = dict(
        seeds=list(range(10)), lanes_per_seed=list(range(4)), minimum_absolute_success_gain=0.15
    )
    baseline = dict(
        high_quality=False,
        clean_foot_only=True,
        minimum_pelvis_z_m=0.7,
        maximum_lateral_excursion_m=2.0,
    )
    rows = [
        dict(
            seed=s,
            lane=lane,
            parent=copy.deepcopy(baseline),
            candidate={**baseline, "high_quality": True},
        )
        for s in range(10)
        for lane in range(4)
    ]
    return rows, protocol


def test_whole_seed_bootstrap_not_forty_independent_lanes():
    rows, protocol = data()
    score = paired_cluster_score(rows, protocol)
    assert score["paired_cluster_ci95"] == [1, 1]
    assert score["independent_seed_cluster_count"] == 10
    assert score["fresh_effect_gate_passed"] is True
    assert score["promotion_authorized"] is False


def test_one_seed_gains_do_not_prove_positive_lower_bound():
    rows, protocol = data()
    for row in rows:
        row["candidate"]["high_quality"] = row["seed"] == 0
    score = paired_cluster_score(rows, protocol)
    assert score["paired_cluster_ci95"][0] == 0
    assert score["fresh_effect_gate_passed"] is False


def test_success_improvement_does_not_override_contact_safety():
    rows, protocol = data()
    rows[0]["candidate"]["clean_foot_only"] = False
    score = paired_cluster_score(rows, protocol)
    assert score["fresh_effect_gate_passed"] is True
    assert score["fresh_safety_gate_passed"] is False


def test_duplicate_missing_or_nonfinite_course_rejected():
    rows, protocol = data()
    rows[-1] = rows[0]
    with pytest.raises(ValueError):
        paired_cluster_score(rows, protocol)
    rows, protocol = data()
    rows[0]["candidate"]["minimum_pelvis_z_m"] = float("nan")
    with pytest.raises(ValueError):
        paired_cluster_score(rows, protocol)
