from copy import deepcopy

import pytest

from rosclaw_soccer.rsi.foot_teacher_pair_comparison import compare_foot_teacher_reviews


def review():
    arm = dict(
        precise_healthy_return=False,
        slew_exceeding_controls=0,
        max_total_joint_jump_rad=0.1,
        outcome=dict(strict_both_clean_handoff=False, both_bodies_healthy=True),
    )
    return dict(
        status="RECEIVER_FOOT_TEACHER_REPLAY_REVIEWED",
        independent_fresh=False,
        promotion_authorized=False,
        zero_identity_count=8,
        prior_hash="same",
        records=[
            dict(
                context=i,
                side=side,
                zero_teacher_matches_lateral_baseline=True,
                baseline=deepcopy(arm),
                teacher=deepcopy(arm),
            )
            for i in range(4)
            for side in (-0.35, 0.35)
        ],
    )


def test_comparison_retains_gains_losses_and_matched_count():
    old = review()
    new = deepcopy(old)
    new["records"][0]["teacher"]["outcome"]["both_bodies_healthy"] = False
    new["records"][1]["teacher"]["max_total_joint_jump_rad"] = 0.02
    result = compare_foot_teacher_reviews(old, new)
    assert result["conditions"] == 8
    assert result["summary"]["healthy"]["lost"] == 1
    assert result["summary"]["max_total_joint_jump_change_rad"]["decreased"] == 1
    assert not result["independent_fresh"] and not result["promotion_authorized"]


@pytest.mark.parametrize("fault", ["parent", "duplicate", "baseline", "nan"])
def test_unmatched_or_invalid_evidence_is_rejected(fault):
    old = review()
    new = deepcopy(old)
    if fault == "parent":
        new["prior_hash"] = "different"
    elif fault == "duplicate":
        new["records"][0] = deepcopy(new["records"][1])
    elif fault == "baseline":
        new["records"][0]["baseline"]["outcome"]["both_bodies_healthy"] = False
    else:
        new["records"][0]["teacher"]["max_total_joint_jump_rad"] = float("nan")
    with pytest.raises(ValueError):
        compare_foot_teacher_reviews(old, new)
