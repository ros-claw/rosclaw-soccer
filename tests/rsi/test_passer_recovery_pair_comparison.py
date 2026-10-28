"""A paired comparison never treats missing or unequal evidence as success."""

from copy import deepcopy

import pytest

from rosclaw_soccer.rsi.passer_recovery_pair_comparison import compare_passer_recovery_reviews


def review():
    records = []
    for i in range(8):
        records.append(
            dict(
                context=i // 2,
                side=(-1, 1)[i % 2],
                baseline_matches_prior=True,
                pre_switch_prefix_exact=True,
                full_prefix_available=True,
                baseline_outcome={"strict_both_clean_handoff": False},
                baseline_precise=False,
                boundary_control=100,
                boundary_target_jump_rad=1.0,
                transition_window_max_target_jump_rad=1.1,
                candidate_outcome={"strict_both_clean_handoff": False},
                candidate_precise=False,
                health=dict(raw_controls=400, executed_controls=500, executed_healthy_full=True),
            )
        )
    return dict(
        status="PASSER_VELOCITY_RECOVERY_REPLAY_REVIEWED",
        independent_fresh=False,
        promotion_authorized=False,
        hardware_authorized=False,
        prior_hash="same",
        records=records,
    )


def test_matched_comparison_reports_all_metrics_and_different_exposure():
    old = review()
    new = deepcopy(old)
    for row in new["records"]:
        row["boundary_target_jump_rad"] = 0.2
        row["transition_window_max_target_jump_rad"] = 0.5
    new["records"][0]["health"]["executed_controls"] = 499
    result = compare_passer_recovery_reviews(old, new)
    assert result["summary"]["boundary_target_jump_rad"]["decreased"] == 8
    assert result["summary"]["transition_window_max_target_jump_rad"]["decreased"] == 8
    assert result["summary"]["health_equal_exposure"]["pairs"] == 7
    assert result["summary"]["health_equal_exposure"]["unequal_pairs"] == 1
    assert result["goal_complete"] is False


@pytest.mark.parametrize("failure", ["prior", "baseline", "prefix", "duplicate", "nan", "health"])
def test_comparison_rejects_incommensurate_or_invalid_evidence(failure):
    old, new = review(), review()
    row = new["records"][0]
    if failure == "prior":
        new["prior_hash"] = "other"
    elif failure == "baseline":
        row["baseline_precise"] = True
    elif failure == "prefix":
        row["pre_switch_prefix_exact"] = False
    elif failure == "duplicate":
        new["records"][1] = deepcopy(row)
    elif failure == "nan":
        row["boundary_target_jump_rad"] = float("nan")
    else:
        row["health"]["executed_healthy_full"] = 1
    with pytest.raises(ValueError):
        compare_passer_recovery_reviews(old, new)
