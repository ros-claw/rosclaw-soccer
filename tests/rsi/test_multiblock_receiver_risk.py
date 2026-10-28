import pytest

from scripts.rsi_train_multiblock_receiver_risk import assemble


def block(contexts, snapshot):
    pairs, features = [], []
    for index in range(24):
        context = contexts[index // 6]
        pairs.append(
            dict(
                index=index,
                context=context,
                baseline_ready=index == 0,
                candidate_ready=index == 0,
                baseline_body_failure=False,
                candidate_body_failure=False,
                baseline_controls=460,
                candidate_controls=460,
            )
        )
        features.append(
            dict(index=index, context=context, observation=[0.0] * 98, ball_feature=[0.0] * 6)
        )
    shared = dict(
        snapshot_hash=snapshot,
        independent_fresh=False,
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
    )
    return (
        dict(status="LONG_RECEIVER_PAIRED_SIM_DIAGNOSIS_NOT_POSSESSION", rows=pairs, **shared),
        dict(status="CAUSAL_RECEIVER_FEATURES_DEV_ONLY_NOT_TRAINED", rows=features, **shared),
    )


def test_join_uses_composite_identity_and_censors_incomplete():
    a, b = block((2, 3, 5, 7), "a"), block((8, 9, 11, 13), "b")
    x, ball, groups, _, _, observed, ids = assemble([a, b])
    assert x.shape == (48, 98) and ball.shape == (48, 6)
    assert observed.sum() == 2 and ids[0]["index"] == ids[24]["index"] == 0
    assert ids[0]["snapshot_hash"] != ids[24]["snapshot_hash"]
    with pytest.raises(ValueError, match="disjoint"):
        assemble([a, a])
    b[1]["snapshot_hash"] = "wrong"
    with pytest.raises(ValueError, match="same-snapshot"):
        assemble([a, b])
