from copy import deepcopy

import pytest

from rosclaw_soccer.rsi.leg_projection_memory_comparison import compare_projection_memory


def review():
    values = dict(
        precise_healthy_return=False,
        outcome=dict(
            strict_both_clean_handoff=False,
            both_bodies_healthy=True,
            passer_foot_precedes_receiver=True,
        ),
    )
    return dict(
        status="RECEIVER_LEG_PROJECTION_REPLAY_REVIEWED",
        independent_fresh=False,
        promotion_authorized=False,
        zero_identity_count=8,
        prior_hash="same",
        hardware_authorized=False,
        raw_action_memory_preserved=True,
        candidate_action_memory_reflects_projection=False,
        records=[
            dict(
                context=i,
                side=s,
                zero_projection_matches_prior=True,
                baseline=deepcopy(values),
                projected=deepcopy(values),
            )
            for i in range(4)
            for s in (-0.35, 0.35)
        ],
    )


def test_matched_memory_comparison_preserves_losses():
    old, new = review(), review()
    new["raw_action_memory_preserved"] = False
    new["candidate_action_memory_reflects_projection"] = True
    new["records"][0]["projected"]["outcome"]["both_bodies_healthy"] = False
    result = compare_projection_memory(old, new)
    assert result["conditions"] == 8 and result["summary"]["healthy"]["lost"] == 1
    assert not result["summary"]["common_exposure_available"]
    assert not result["independent_fresh"] and not result["promotion_authorized"]


def test_common_horizon_does_not_count_later_failure_as_a_loss():
    old, new = review(), review()
    new["raw_action_memory_preserved"] = False
    new["candidate_action_memory_reflects_projection"] = True
    for report in (old, new):
        for row in report["records"]:
            row["projected"]["observed_controls"] = 300
            row["projected"]["first_body_failure_controls"] = [None, None]
    candidate = new["records"][0]["projected"]
    candidate["observed_controls"] = 520
    candidate["first_body_failure_controls"] = [519, None]
    candidate["outcome"]["both_bodies_healthy"] = False
    summary = compare_projection_memory(old, new)["summary"]
    assert summary["healthy"]["lost"] == 1
    assert summary["unequal_exposure_pairs"] == 1
    assert summary["healthy_common"]["lost"] == 0


def test_partial_timeline_is_rejected():
    old, new = review(), review()
    new["raw_action_memory_preserved"] = False
    new["candidate_action_memory_reflects_projection"] = True
    old["records"][0]["projected"]["observed_controls"] = 300
    with pytest.raises(ValueError):
        compare_projection_memory(old, new)


@pytest.mark.parametrize("fault", ["parent", "identity", "baseline", "duplicate", "boolean"])
def test_reject_unmatched_or_invalid_evidence(fault):
    old, new = review(), review()
    new["raw_action_memory_preserved"] = False
    new["candidate_action_memory_reflects_projection"] = True
    if fault == "parent":
        new["prior_hash"] = "different"
    elif fault == "identity":
        new["zero_identity_count"] = 7
    elif fault == "baseline":
        new["records"][0]["baseline"]["outcome"]["both_bodies_healthy"] = False
    elif fault == "duplicate":
        new["records"][0] = deepcopy(new["records"][1])
    else:
        new["records"][0]["projected"]["outcome"]["both_bodies_healthy"] = 1
    with pytest.raises(ValueError):
        compare_projection_memory(old, new)
