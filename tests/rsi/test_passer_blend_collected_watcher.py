import json

import pytest

from scripts import rsi_watch_passer_blend_comparison as watcher


def test_collected_worker_requires_bound_successful_replay(tmp_path, monkeypatch):
    review = tmp_path / "review" / "complete.json"
    review.parent.mkdir()
    receipt = dict(
        status="PASSER_VELOCITY_RECOVERY_REPLAY_REVIEWED",
        source_hash="source",
        audit_hash="audit",
        source_head="reviewer",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    review.write_text(json.dumps(receipt))
    monkeypatch.setattr(
        watcher, "report", lambda *a, **kw: ({}, {"source_head": "reviewer"}, "source")
    )
    collected = {"LoadState": "not-found", "ActiveState": "inactive"}
    assert watcher.collected_replay_verified(collected, review)
    assert not watcher.collected_replay_verified({"LoadState": "loaded"}, review)
    receipt["source_hash"] = "different"
    review.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="bind exact"):
        watcher.collected_replay_verified(collected, review)


def test_missing_review_never_counts_as_success(tmp_path):
    with pytest.raises(ValueError, match="completed review"):
        watcher.collected_replay_verified(
            {"LoadState": "not-found", "ActiveState": "inactive"}, tmp_path / "missing.json"
        )
