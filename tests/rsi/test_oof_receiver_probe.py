import numpy as np
import pytest

from rosclaw_soccer.rsi.oof_receiver_probe import select_oof_probe


def receipt():
    groups = (2, 3, 5, 7, 8, 9, 11, 13, 15, 16, 17, 18)
    identities = [dict(snapshot_hash=f"s{g}", index=i, context=g) for g in groups for i in range(6)]
    models = [
        dict(
            group=g,
            kind="linear",
            mean=[0.0] * 104,
            scale=[1.0] * 104,
            weight=[0.0] * 104,
            bias=0.0,
        )
        for g in groups
    ]
    return dict(
        status="MULTIBLOCK_GROUPED_RISK_RESEARCH_NOT_PHYSICALLY_VALIDATED",
        source_head="c0a2b4cd2a9c15bc6f2f5097bedd923808a1af6a",
        exported_fold_models=True,
        predicted_unobserved=True,
        threshold=0.8,
        independent_fresh=False,
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
        identities=identities,
        ball_augmented=dict(fold_models=models, probabilities=[0.5] * 72, predictions=[0] * 72),
    )


def test_decision_is_computed_from_current_state_and_audited():
    data = receipt()
    q, v = np.zeros(43), np.zeros(41)
    q[3] = 1
    obs = np.zeros(98, np.float32)
    kwargs = dict(
        snapshot_hash="s2", index=0, context=2, observation=obs, q=q, v=v, variant="ball_augmented"
    )
    assert select_oof_probe(data, **kwargs) == (0, 0.5)
    data["ball_augmented"]["predictions"][0] = 1
    with pytest.raises(ValueError, match="differs"):
        select_oof_probe(data, **kwargs)
