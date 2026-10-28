from scripts.rsi_watch_new_context_proofs import valid_capture


def test_capture_must_cover_the_exact_declared_block_and_lineage():
    contexts = (8, 9, 11, 13)
    rows = [
        dict(index=i, context=c, side=s, receiver_lateral_offset_m=o)
        for i, (c, s, o) in enumerate(
            (c, s, o) for c in contexts for s in (-0.35, 0.35) for o in (-0.35, -0.12, 0.35)
        )
    ]
    receipt = dict(
        status="RELAY_NONDEFAULT_PRECONTACT_NATIVE_STATE_RECONSTRUCTION_NOT_POLICY",
        source_head="head",
        source_hash="source",
        audit_hash="audit",
        contexts=list(contexts),
        rows=rows,
        saved_torque_is_not_policy=True,
        independent_fresh=False,
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
    )
    kwargs = dict(head="head", source_hash="source", audit_hash="audit", contexts=contexts)
    assert valid_capture(receipt, **kwargs)
    receipt["rows"][-1] = receipt["rows"][0]
    assert not valid_capture(receipt, **kwargs)
