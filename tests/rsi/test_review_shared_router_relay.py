import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_receiving_handoff_diagnosis import trajectory

from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_review_shared_router_relay import (
    admit,
    optimistic_handoff_diagnosis,
    verify_own_handoff,
    verify_previous_target_feedback,
)


def study(root, *, replay, source_hash=None):
    root.mkdir()
    declaration = root / "declaration.json"
    declaration.write_text(json.dumps(dict(input_file_hashes={}, protocol_hash="protocol")))
    rows = []
    for i in range(16):
        folder = root / f"trial{i}"
        folder.mkdir()
        path = folder / "physics.npz"
        np.savez(path, state=np.array([i], dtype=np.int64))
        rows.append(
            dict(
                index=i,
                arm="dual" if i % 2 == 0 else "router",
                artifacts={"physics": dict(path=f"trial{i}/physics.npz", hash=digest(path))},
            )
        )
    report = dict(
        status="ROUTER_SHARED_RELAY_OWN_REPLAY_EXACT"
        if replay
        else "ROUTER_SHARED_RELAY_SOURCE_PENDING_REPLAY",
        source_hash=source_hash,
        source_head="head",
        worlds=16,
        records=rows,
        declaration_hash=digest(declaration),
        hardware_authorized=False,
        promotion_authorized=False,
        team_game_verified=False,
    )
    path = root / "complete.json"
    path.write_text(json.dumps(report))
    return digest(path)


def test_source_replay_binding_and_array_tamper_rejected(tmp_path):
    source, replay = tmp_path / "source", tmp_path / "replay"
    source_hash = study(source, replay=False)
    replay_hash = study(replay, replay=True, source_hash=source_hash)
    old, _ = admit(source, replay, source_hash, replay_hash)
    assert len(old["records"]) == 16
    np.savez(replay / "trial3/physics.npz", state=np.array([999]))
    with pytest.raises(ValueError, match="artifact"):
        admit(source, replay, source_hash, replay_hash)


def test_replay_must_bind_exact_source_hash(tmp_path):
    source, replay = tmp_path / "source", tmp_path / "replay"
    source_hash = study(source, replay=False)
    replay_hash = study(replay, replay=True, source_hash="wrong")
    with pytest.raises(ValueError, match="16-world"):
        admit(source, replay, source_hash, replay_hash)


def test_handoff_recurrent_and_executed_action_continuity():
    state = np.zeros((3, 2, 1, 128), np.float32)
    shadow = {"h_after": state[:1].copy(), "c_after": state[:1].copy()}
    returned = {
        "NN_h_before": state[:2].copy(),
        "NN_h_after": state[:2].copy(),
        "NN_c_before": state[:2].copy(),
        "NN_c_after": state[:2].copy(),
        "NN_obs": np.zeros((2, 160), np.float32),
        "NN_action": np.zeros((2, 29), np.float32),
    }
    verify_own_handoff(shadow, returned, handoff=1, controls=3, width=128)
    returned["NN_obs"][1, 125] = 1
    with pytest.raises(ValueError, match="committed action"):
        verify_own_handoff(shadow, returned, handoff=1, controls=3, width=128)
    returned["NN_obs"][1, 125] = 0
    returned["NN_h_before"][0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="recurrent"):
        verify_own_handoff(shadow, returned, handoff=1, controls=3, width=128)


def test_shadow_feedback_comes_from_actual_applied_target():
    offset = np.zeros(29, np.float32)
    scale = np.full(29, 0.5, np.float32)
    order = np.arange(29)[::-1]
    targets = np.ones((2, 29), np.float64) * 0.1
    shadow = {"obs": np.zeros((2, 160), np.float32)}
    shadow["obs"][1, 125:154] = 0.2
    returned = {"NN_obs": np.zeros((1, 160), np.float32)}
    returned["NN_obs"][0, 125:154] = 0.2
    verify_previous_target_feedback(
        shadow, returned, targets, handoff=2, offset=offset, scale=scale, order=order
    )
    shadow["obs"][1, 125] = 0
    with pytest.raises(ValueError, match="shadow observation"):
        verify_previous_target_feedback(
            shadow, returned, targets, handoff=2, offset=offset, scale=scale, order=order
        )


def test_optimistic_no_handoff_bound_never_claims_excluded_contact_known():
    q, v, force, failed = trajectory()
    physics = dict(qpos=q, qvel=v, ball_forces=force, body_failed=failed)
    layout = SimpleNamespace(rq=0, rv=0, bq=36, bv=35)
    bound = optimistic_handoff_diagnosis(physics, layout)
    assert bound["actual_excluded_contact_unknown"]
    assert bound["upper_bound_insufficient_for_15_controls"]
    assert "predicate_reconstruction_exact" not in bound["diagnostic"]
    force[285, 4, 1, 0] = 2
    bound = optimistic_handoff_diagnosis(physics, layout)
    assert not bound["upper_bound_insufficient_for_15_controls"]
    assert bound["actual_excluded_contact_unknown"]
