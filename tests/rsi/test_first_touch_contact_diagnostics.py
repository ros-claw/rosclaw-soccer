"""Contact-order diagnostics recompute per-body events from sealed traces."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi import first_touch_contact_diagnostics as diagnostics


def test_nonfoot_removal_is_visible_without_claiming_promotion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent = tmp_path / "parent"
    candidate = tmp_path / "candidate"
    parent.mkdir()
    candidate.mkdir()
    parent_force = np.zeros((5, 1, 6), dtype=np.float32)
    parent_force[2, 0, 0] = 10.0
    parent_force[3, 0, 4] = 2.0
    candidate_force = parent_force.copy()
    candidate_force[3, 0, 4] = 0.0
    np.savez(parent / "trace.npz", ball_body_contact_force_peak_n=parent_force)
    np.savez(candidate / "trace.npz", ball_body_contact_force_peak_n=candidate_force)
    monkeypatch.setattr(
        diagnostics,
        "audit_vector_first_touch",
        lambda _: {"report_hash": "parent-audit", "clean_foot_only_episode_count": 0},
    )
    monkeypatch.setattr(
        diagnostics,
        "audit_first_touch_candidate_execution",
        lambda *_, **__: {"report_hash": "candidate-audit", "candidate_clean_foot_only_count": 1},
    )
    report = diagnostics.diagnose_contact_order(parent, candidate, tmp_path / "manifest.json")
    change = report["course_changes"][0]
    assert change["parent"]["first_nonfoot_frame"] == 3
    assert change["candidate"]["first_nonfoot_frame"] is None
    assert change["nonfoot_removed"] is True
    assert report["promotion_authorized"] is False
