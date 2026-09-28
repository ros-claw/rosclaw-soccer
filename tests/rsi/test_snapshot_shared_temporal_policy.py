"""A learned snapshot actor remains shared, bounded and bank-bound."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.snapshot_replay_evidence import projected_probe_residual
from rosclaw_soccer.rsi.snapshot_shared_temporal_policy import (
    candidate_manifest,
    load_candidate,
)

BANK = "sha256:" + "a" * 64


def test_shared_candidate_roundtrip_and_bank_binding(tmp_path: Path) -> None:
    weights = np.zeros((6, 3))
    weights[0, 0] = 0.5
    data = candidate_manifest(weights, bank_manifest_hash=BANK, seed=9)
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    digest, restored = load_candidate(path, bank_manifest_hash=BANK)
    assert digest == data["candidate_hash"]
    np.testing.assert_array_equal(restored, weights)
    with pytest.raises(ValueError, match="commitment"):
        load_candidate(path, bank_manifest_hash="sha256:" + "b" * 64)


def test_shared_candidate_rejects_tampering_and_unbounded_actions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bounded"):
        candidate_manifest(np.full((6, 3), 1.1), bank_manifest_hash=BANK, seed=1)
    data = candidate_manifest(np.zeros((6, 3)), bank_manifest_hash=BANK, seed=1)
    data["weights"][0][1] = 0.2
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="commitment"):
        load_candidate(path, bank_manifest_hash=BANK)


def test_shared_candidate_audit_recomputes_joint_limit_projection() -> None:
    baseline = np.zeros(29)
    limits = np.asarray([[-1.0, 1.0], [-1.0, 1.0], [-0.5, 0.5]])
    baseline[2] = 0.5
    actual = projected_probe_residual(np.asarray([0.02, -0.03, 0.04]), baseline, limits, [0, 1, 2])
    np.testing.assert_allclose(actual, [0.02, -0.03, 0.0])
