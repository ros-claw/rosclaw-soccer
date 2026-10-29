import json

import numpy as np
import pytest

from scripts.rsi_native_context_recovery_common import digest
from scripts.rsi_review_shared_router_relay import admit


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
