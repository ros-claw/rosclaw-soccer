from scripts.rsi_watch_oof_physical_review import valid_candidate


def test_requires_all_24_actual_oof_decisions_and_no_hardware():
    d = dict(
        status="RELAY_OOF_RISK_PHYSICAL_DEVELOPMENT_NOT_SUCCESS",
        source_head="h",
        snapshot_hash="s",
        risk_receipt_hash="r",
        risk_variant="ball_augmented",
        rows=[dict(index=i, selector_class=0, risk_probability=0.5) for i in range(24)],
        future_controls_used=False,
        independent_fresh=False,
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
    )
    args = dict(source_head="h", snapshot_hash="s", risk_hash="r")
    assert valid_candidate(d, **args)
    d["rows"][0]["selector_class"] = None
    assert not valid_candidate(d, **args)
