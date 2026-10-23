import copy

import pytest

from rosclaw_soccer.rsi.pass_transfer_learning import (
    controlled_pass_label,
    paired_pass_learning_summary,
)


def outcome():
    return dict(
        whole_body_healthy=True,
        passer_full100NNrecovery=True,
        passer_clean_foot=True,
        B_legal_functional_receive_before_deadline=True,
        B_excluded_contact=False,
        incoming=dict(A_first_foot_strike_substep=100, B_first_ball_contact_substep=1000),
        combined_precision_and_functional_receive=False,
    )


def test_training_label_requires_real_A_launch_before_B_and_full_recovery():
    o = outcome()
    assert controlled_pass_label(o)
    for k in (
        "whole_body_healthy",
        "passer_full100NNrecovery",
        "passer_clean_foot",
        "B_legal_functional_receive_before_deadline",
    ):
        bad = copy.deepcopy(o)
        bad[k] = False
        assert not controlled_pass_label(bad)
    for first in (None, 1000, 1001):
        bad = copy.deepcopy(o)
        bad["incoming"]["A_first_foot_strike_substep"] = first
        assert not controlled_pass_label(bad)
    o["B_excluded_contact"] = True
    assert not controlled_pass_label(o)


def rows():
    result = []
    for state in range(40):
        for task in (0, 1):
            for arm in (False, True):
                o = outcome()
                o["B_legal_functional_receive_before_deadline"] = False
                result.append(
                    dict(
                        course=dict(
                            state_id=state,
                            task=task,
                            assisted=arm,
                            context=(1, 4, 20, 21)[state // 10],
                        ),
                        outcome=o,
                    )
                )
    return result


def test_paired_rescues_and_losses_keep40_state_clusters_and_static_negatives():
    records = rows()
    for i in (0, 5, 9):
        records[i]["outcome"]["B_legal_functional_receive_before_deadline"] = True
    s = paired_pass_learning_summary(records)
    assert s["baseline_controlled"] == 1 and s["teacher_controlled"] == 2
    assert s["paired_net_change"] == 1 and len(s["rescues"]) == 2 and len(s["regressions"]) == 1
    assert s["executions"] == 160 and s["distinct_physical_initial_conditions"] == 40
    assert s["source_context_clusters"] == 4 and s["strict_static_point_composite_passes"] == 0
    assert not s["strict_point_exam_regraded"] and not s["goal_complete"]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "context", "bool"])
def test_no_partial_or_invalid_bank_summary(fault):
    records = rows()
    if fault == "missing":
        records.pop()
    elif fault == "duplicate":
        records[1] = copy.deepcopy(records[0])
    elif fault == "context":
        records[0]["course"]["context"] = 999
    else:
        records[0]["outcome"]["whole_body_healthy"] = 1
    with pytest.raises(ValueError):
        paired_pass_learning_summary(records)
