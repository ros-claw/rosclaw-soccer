import json
import os
import shutil
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.verify_sonic_contact_holdout import verify_holdout
from rosclaw_soccer.sim.contracts import hash_json


@pytest.fixture
def holdout_copy(tmp_path):
    source = os.environ.get("ROSCLAW_SOCCER_RSI_SELECTOR_HOLDOUT")
    stadium = os.environ.get("ROSCLAW_SOCCER_RSI_SELECTOR_STADIUM")
    if not source or not stadium:
        pytest.skip("external selector holdout evidence was not provided")
    return shutil.copytree(Path(source), tmp_path / "holdout"), Path(stadium)


def test_selector_holdout_recomputes_eight_physical_outcomes(holdout_copy):
    root, stadium = holdout_copy
    result = verify_holdout(root, stadium_assets=stadium)
    assert result["physical_execution_count"] == 8
    assert result["candidate_foot_goals"] == 4
    assert result["parent_foot_goals"] == 2
    assert result["contact_independently_reconstructed"]
    assert not result["promotion_authorized"]


def test_selector_holdout_rejects_raw_trace_tamper(holdout_copy):
    root, stadium = holdout_copy
    trace = root / "x1800-y0040-candidate" / "trajectory.npz"
    trace.write_bytes(trace.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="trajectory hash mismatch"):
        verify_holdout(root, stadium_assets=stadium)


def test_selector_holdout_rejects_forged_totals(holdout_copy):
    root, stadium = holdout_copy
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.pop("manifest_hash")
    manifest["candidate_foot_goals"] = 0
    manifest["manifest_hash"] = hash_json(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="totals"):
        verify_holdout(root, stadium_assets=stadium)
