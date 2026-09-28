"""Fail-closed tests for immutable SIM_ONLY first-touch snapshot evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.first_touch_snapshot_bank import (
    SCHEMA,
    _extract,
    audit_snapshot_bank,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


@pytest.mark.parametrize(
    ("lead", "window"),
    [(14, 80), (80, 80), (81, 80), (45, 151), (True, 80), (30, False)],
)
def test_snapshot_horizon_rejects_invalid_values(lead: int, window: int) -> None:
    with pytest.raises(ValueError, match="horizon"):
        _extract((Path("/nonexistent-source"),), lead_frames=lead, window_frames=window)


@pytest.mark.parametrize("start", [-1, 0, True, 2.5])
def test_snapshot_horizon_rejects_invalid_common_start(start: int) -> None:
    with pytest.raises(ValueError, match="horizon"):
        _extract((Path("/nonexistent-source"),), fixed_start_frame=start)


def test_snapshot_bank_rejects_modified_archive_before_source_rebuild(tmp_path: Path) -> None:
    archive = tmp_path / "snapshots.npz"
    archive.write_bytes(b"sealed-simulated-state")
    manifest = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "learning_authorized": False,
        "promotion_authorized": False,
        "privileged_future_targets_diagnostic_only": True,
        "archive_hash": hash_bytes(archive.read_bytes()),
        "snapshots": [],
    }
    manifest["manifest_hash"] = hash_json(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    archive.write_bytes(b"modified-simulated-state")
    with pytest.raises(ValueError, match="unauthenticated"):
        audit_snapshot_bank(tmp_path)


def test_snapshot_bank_rejects_promotion_flag(tmp_path: Path) -> None:
    archive = tmp_path / "snapshots.npz"
    archive.write_bytes(b"sealed-simulated-state")
    manifest = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "learning_authorized": False,
        "promotion_authorized": True,
        "privileged_future_targets_diagnostic_only": True,
        "archive_hash": hash_bytes(archive.read_bytes()),
        "snapshots": [],
    }
    manifest["manifest_hash"] = hash_json(manifest)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unauthenticated"):
        audit_snapshot_bank(tmp_path)
