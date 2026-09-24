import json
import os
import shutil
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.verify_sonic_combine import verify_sonic_combine
from rosclaw_soccer.sim.contracts import hash_json


@pytest.fixture
def evidence(tmp_path):
    source = os.environ.get("ROSCLAW_SOCCER_RSI_COMBINE_EVIDENCE")
    if source is None:
        pytest.skip("external SONIC Combine evidence was not supplied")
    path = Path(source)
    if not (path / "complete.json").is_file():
        pytest.skip("external SONIC Combine evidence is unavailable")
    return shutil.copytree(path, tmp_path / "evidence")


def test_real_sonic_combine_bundle_has_replayed_physical_metrics(evidence):
    result = verify_sonic_combine(evidence)
    assert result["pass_count"] == 5
    assert result["physical_execution_count"] == 11
    assert result["causal_separation"]
    assert not result["promotion_authorized"]


def test_sonic_combine_rejects_tampered_raw_trace(evidence):
    trace = evidence / "walk-primary.npz"
    trace.write_bytes(trace.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="trajectory hash mismatch"):
        verify_sonic_combine(evidence)


def test_sonic_combine_rejects_reforged_promotion(evidence):
    path = evidence / "complete.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt.pop("manifest_hash")
    receipt["promotion_authorized"] = True
    receipt["manifest_hash"] = hash_json(receipt)
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="boundary mismatch"):
        verify_sonic_combine(evidence)
