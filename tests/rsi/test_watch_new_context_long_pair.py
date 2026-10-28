from scripts.rsi_watch_new_context_long_pair import valid_dual


def test_dual_gate_requires_every_case_and_exact_physical_steps():
    contexts = (8, 9, 11, 13)
    rows = [
        dict(
            index=i,
            context=c,
            side=s,
            receiver_lateral_offset_m=o,
            consecutive_controls=20,
            coupled_native_substeps_exact=True,
        )
        for i, (c, s, o) in enumerate(
            (c, s, o) for c in contexts for s in (-0.35, 0.35) for o in (-0.35, -0.12, 0.35)
        )
    ]
    proof = dict(
        status="RELAY_NONDEFAULT_DUAL_NEURAL_20_CONTROLS_MATCH_NOT_FOOTBALL_SUCCESS",
        source_hash="s",
        audit_hash="a",
        snapshot_hash="p",
        expanded_velocity_proof_hash="v",
        coupled_physics_continuation_verified=True,
        native_steps=4800,
        rows=rows,
        independent_fresh=False,
        hardware_authorized=False,
        promotion_authorized=False,
        goal_complete=False,
        future_controls_used=False,
    )
    kwargs = dict(
        source_hash="s", audit_hash="a", snapshot_hash="p", velocity_hash="v", contexts=contexts
    )
    assert valid_dual(proof, **kwargs)
    proof["rows"][0]["coupled_native_substeps_exact"] = False
    assert not valid_dual(proof, **kwargs)
